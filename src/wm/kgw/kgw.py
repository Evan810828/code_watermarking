"""KGW watermark (Kirchenbauer et al., 2023, "A Watermark for Large Language Models").

Implementation follows MarkLLM's watermark/kgw
(https://github.com/THU-BPM/MarkLLM/tree/main/watermark/kgw):

- At each decoding step, a pseudo-random "green list" of gamma*|V| tokens is
  derived from the previous `prefix_length` tokens (seeded with `hash_key`).
- Generation: add `delta` to the logits of green-list tokens (KGWLogitsProcessor).
- Detection: re-derive the green list at every position and count green hits;
  z = (n_green - gamma*T) / sqrt(T * gamma * (1-gamma)), flag if z > z_threshold.

Default hyperparameters mirror MarkLLM's config/KGW.json:
gamma=0.5, delta=2.0, hash_key=15485863, prefix_length=1, z_threshold=4.0,
f_scheme="time", window_scheme="left".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
from transformers import LogitsProcessor


@dataclass
class KGWConfig:
    vocab_size: int
    gamma: float = 0.5
    delta: float = 2.0
    hash_key: int = 15485863
    prefix_length: int = 1
    z_threshold: float = 4.0
    f_scheme: str = "time"        # time | additive | skip | min
    window_scheme: str = "left"   # left | self
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


class KGWUtils:
    """Green-list derivation and sequence scoring (mirrors MarkLLM KGWUtils)."""

    def __init__(self, config: KGWConfig):
        self.config = config
        self.rng = torch.Generator(device=config.device)
        self.rng.manual_seed(config.hash_key)
        # Fixed pseudo-random permutation of the vocab used by the f-schemes.
        self.prf = torch.randperm(
            config.vocab_size, device=config.device, generator=self.rng
        )
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min,
        }
        self.window_scheme_map = {
            "left": self._get_greenlist_ids_left,
            "self": self._get_greenlist_ids_self,
        }

    def _f(self, input_ids: torch.LongTensor) -> int:
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))

    def _f_time(self, input_ids: torch.LongTensor) -> int:
        time_result = 1
        for i in range(self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]

    def _f_additive(self, input_ids: torch.LongTensor) -> int:
        additive_result = 0
        for i in range(self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]

    def _f_skip(self, input_ids: torch.LongTensor) -> int:
        return self.prf[input_ids[-self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor) -> int:
        return min(
            self.prf[input_ids[-1 - i].item()]
            for i in range(self.config.prefix_length)
        )

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Green list for the *next* token given the context `input_ids`."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        self.rng.manual_seed((self.config.hash_key * self._f(input_ids)) % (2**63 - 1))
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(
            self.config.vocab_size, device=self.config.device, generator=self.rng
        )
        return vocab_permutation[:greenlist_size]

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        f_x = self._f(input_ids)
        candidates = (self.prf * f_x) % self.config.vocab_size
        return torch.nonzero(candidates < greenlist_size, as_tuple=True)[0]

    def compute_z_score(self, observed_count: int, T: int) -> float:
        expected = self.config.gamma
        numer = observed_count - expected * T
        denom = math.sqrt(T * expected * (1 - expected))
        return numer / denom

    def score_sequence(
        self, input_ids: torch.LongTensor, scored_from: int | None = None
    ) -> tuple[float, int, int, list[int]]:
        """Score tokens input_ids[scored_from:] using their true left context.

        `scored_from` defaults to `prefix_length` (detect on standalone text).
        Pass the prompt length to score only generated tokens with the exact
        context seen at generation time.

        Returns (z_score, green_count, num_scored, green_flags).
        """
        if scored_from is None:
            scored_from = self.config.prefix_length
        scored_from = max(scored_from, self.config.prefix_length)
        num_scored = len(input_ids) - scored_from
        if num_scored < 1:
            return 0.0, 0, 0, []
        green_count = 0
        green_flags: list[int] = []
        for idx in range(scored_from, len(input_ids)):
            greenlist = self.get_greenlist_ids(input_ids[:idx])
            if int(input_ids[idx]) in set(greenlist.tolist()):
                green_count += 1
                green_flags.append(1)
            else:
                green_flags.append(0)
        z = self.compute_z_score(green_count, num_scored)
        return z, green_count, num_scored, green_flags


class KGWLogitsProcessor(LogitsProcessor):
    """Adds `delta` to green-list token logits at every decoding step."""

    def __init__(self, config: KGWConfig, utils: KGWUtils | None = None):
        self.config = config
        self.utils = utils if utils is not None else KGWUtils(config)

    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, batched_greenlist_ids: list[torch.LongTensor]
    ) -> torch.BoolTensor:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, ids in enumerate(batched_greenlist_ids):
            mask[b_idx, ids.to(scores.device)] = True
        return mask

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        batched_greenlist_ids = [
            self.utils.get_greenlist_ids(input_ids[b_idx])
            for b_idx in range(input_ids.shape[0])
        ]
        green_mask = self._calc_greenlist_mask(scores, batched_greenlist_ids)
        scores = scores.clone()
        scores[green_mask] = scores[green_mask] + self.config.delta
        return scores


class KGWDetector:
    """z-test detector over token ids."""

    def __init__(self, config: KGWConfig, utils: KGWUtils | None = None):
        self.config = config
        self.utils = utils if utils is not None else KGWUtils(config)

    def detect(
        self, input_ids: torch.LongTensor, scored_from: int | None = None
    ) -> dict:
        z, green_count, num_scored, _ = self.utils.score_sequence(
            input_ids, scored_from=scored_from
        )
        return {
            "z_score": z,
            "green_token_count": green_count,
            "num_tokens_scored": num_scored,
            "green_fraction": green_count / num_scored if num_scored else None,
            "is_watermarked": bool(z > self.config.z_threshold),
        }
