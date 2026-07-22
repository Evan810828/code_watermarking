"""Code watermark of Guo & Cheng (NeurIPS 2025, "Practical and Effective
Code Watermarking for Large Language Models", ../code-watermark). Core-only
port of models/code.py + the models/wllm.py base pieces it builds on.

Algorithm (`--wm code` upstream): at each step, seed an RNG from
hash_key * previous_token (KGW simple_1), run the trained CodeModel on the
last `context_width` tokens, sort its vocab logits descending, pair adjacent
ranks (0,1)(2,3)..., and draw one member of every pair as the greenlist —
so the greenlist is half the vocab (z-test gamma 0.5) and, because the
CodeModel is distilled from the teacher, each pair holds two roughly
equiprobable continuations: biasing one of them steers sampling without
degrading quality the way a random KGW greenlist does. The CodeModel's
sigmoid "switch" head gates both embedding and scoring to positions it
predicts as high-entropy (the SWEET idea, but without needing the teacher
LLM at detection time).

Detection scores the completion token ids alone: every
(context_width+1)-gram whose switch exceeds switch_threshold contributes one
Bernoulli(gamma) trial to the z-test.

Port deltas (documented; none change the algorithm):
  - upstream code.py:263 reads `len(text) < self.context_width <+1`, a
    chained-comparison typo that never fires; restored to the intended
    `< self.context_width + 1` invalid-input guard.
  - the pair-choice RNG is a CPU torch.Generator in both processor and
    detector (upstream seeds a generator on the model's device; CPU makes
    greenlists device-independent, and both sides of every run go through
    this port).
  - the CodeModel is always called with a single [1, context_width] input in
    both processor and detector — same shape upstream — so both sides see
    bit-identical logits (a batched call could reorder float reductions and
    flip near-tied ranks, silently corrupting detection).
  - detector's per-ngram switch filter uses a set (upstream builds a list
    and does O(n^2) membership tests); output identical.
  - detection dicts carry this repo's unified keys (z_score /
    is_watermarked) alongside the upstream ones (prediction, confidence),
    like wm/sweet.
  - detector drops the tokenizer/BOS-strip plumbing: runners pass raw
    generated ids, which never start with a BOS token (Qwen has none).
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import sqrt

import scipy.stats
import torch
from transformers import LogitsProcessor


@dataclass
class CodeWmConfig:
    vocab_size: int
    gamma: float = 0.5  # pair selection fixes the green fraction at 1/2
    delta: float = 2.0  # upstream run.sh default
    hash_key: int = 15485863
    context_width: int = 2  # upstream README gen_mbpp recipe
    switch_threshold: float = 0.7  # upstream README run example
    entropy_threshold: float = 1.2  # switch's training target (provenance)
    z_threshold: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


def ngrams(sequence: list, n: int):
    """Sliding n-grams (upstream src/datahelper.py ngrams, no padding)."""
    iterables = [sequence[i:] for i in range(n)]
    return zip(*iterables)


class CodeWmBase:
    """Greenlist construction shared by processor and detector (CodeMixin +
    WatermarkBase seeding)."""

    def __init__(self, config: CodeWmConfig, model):
        self.config = config
        self.model = model
        self.model.eval()
        self.rng = torch.Generator()  # CPU (see port deltas)

    @torch.no_grad()
    def _get_greenlist_ids(self, input_ids: torch.LongTensor):
        """(greenlist ids [~vocab/2], switch in (0,1)) for one context.
        input_ids: 1-D, at least context_width tokens; the RNG is seeded
        from the LAST token (simple_1), the CodeModel reads the last
        context_width tokens."""
        cw = self.config.context_width
        if input_ids.shape[-1] < cw:
            raise ValueError(f"need a {cw}-token prefix to seed the greenlist")
        self.rng.manual_seed(self.config.hash_key * int(input_ids[-1]))

        device = next(self.model.parameters()).device
        ctx = input_ids[-cw:].to(device)
        output = self.model(ctx)  # single [1, cw] call, see port deltas
        logits, switch = output["logits"], output["switch"]
        switch = float(torch.sigmoid(switch).squeeze(0))

        sorted_indices = torch.sort(logits, dim=1, descending=True).indices.squeeze(0)
        seq_len = sorted_indices.shape[0]
        k = (seq_len + 1) // 2
        random_choices = torch.randint(0, 2, (k,), generator=self.rng)

        base_indices = torch.arange(0, seq_len - 1, 2)
        selected_indices = base_indices + random_choices[: len(base_indices)]
        if seq_len % 2 == 1 and random_choices[-1].item():
            selected_indices = torch.cat(
                [selected_indices, torch.tensor([seq_len - 1])]
            )
        greenlist_ids = sorted_indices[selected_indices.to(device)]
        return greenlist_ids, switch


class CodeLogitsProcessor(CodeWmBase, LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        switch_list = []
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids, switch = self._get_greenlist_ids(input_ids[b_idx])
            green_tokens_mask[b_idx][greenlist_ids.to(scores.device)] = True
            switch_list.append(switch)

        switch_mask = (
            torch.tensor(switch_list, device=scores.device)
            > self.config.switch_threshold
        ).view(-1, 1)
        green_tokens_mask = green_tokens_mask * switch_mask
        scores[green_tokens_mask] = scores[green_tokens_mask] + self.config.delta
        return scores


class CodeDetector(CodeWmBase):
    @lru_cache(maxsize=2**20)
    def _get_ngram_score_cached(self, prefix: tuple, target: int):
        greenlist_ids, switch = self._get_greenlist_ids(torch.as_tensor(prefix))
        green = bool((greenlist_ids == target).any())
        return green, switch

    def _score_sequence(self, input_ids: torch.Tensor) -> dict:
        cw = self.config.context_width
        token_ngrams = collections.Counter(ngrams(input_ids.tolist(), cw + 1))
        green_lookup, switch_lookup = {}, {}
        for ngram in token_ngrams:
            green, switch = self._get_ngram_score_cached(ngram[:-1], ngram[-1])
            green_lookup[ngram] = green
            switch_lookup[ngram] = switch

        # new_count_scores (non-ignore_repeated_ngrams branch): only ngrams
        # whose switch clears the threshold are scored
        gated = {n for n, s in switch_lookup.items() if s > self.config.switch_threshold}
        num_tokens_scored = sum(f for n, f in token_ngrams.items() if n in gated)
        green_token_count = sum(
            f * green_lookup[n] for n, f in token_ngrams.items() if n in gated
        )
        num_tokens_total = sum(token_ngrams.values())

        if num_tokens_scored == 0:
            return {
                "num_tokens_total": num_tokens_total,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "green_fraction": 0.0,
                "switch_fraction": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
            }

        g = self.config.gamma
        z = (green_token_count - g * num_tokens_scored) / sqrt(
            num_tokens_scored * g * (1 - g)
        )
        return {
            "num_tokens_total": num_tokens_total,
            "num_tokens_scored": num_tokens_scored,
            "num_green_tokens": green_token_count,
            "green_fraction": green_token_count / num_tokens_scored,
            "switch_fraction": num_tokens_scored / num_tokens_total,
            "z_score": z,
            "p_value": float(scipy.stats.norm.sf(z)),
        }

    def detect(self, tokenized_text: torch.Tensor) -> dict:
        tokenized_text = tokenized_text.cpu()
        if len(tokenized_text) < self.config.context_width + 1:
            return {"invalid": True, "z_score": None, "is_watermarked": False}

        out = self._score_sequence(tokenized_text)
        prediction = bool(out["z_score"] > self.config.z_threshold)
        out["prediction"] = prediction
        out["is_watermarked"] = prediction
        if prediction:
            out["confidence"] = 1 - out["p_value"]
        return out
