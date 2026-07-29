"""ACW-s — the paper's no-WPN variant (upstream `--wm sweetcode`,
models/sweetcode.py): logits-guided sampling without the watermark
prediction network.

At every generation step the candidate pairing comes from sorting the LLM's
OWN current logits (descending; adjacent ranks paired (0,1)(2,3)...; a
seeded coin picks the green member of each pair — exactly CodeWM's pairing,
but over the true next-token distribution instead of the distilled
CodeModel), and the gate is the TRUE Shannon entropy of that distribution
(> entropy_threshold) rather than the learned switch. Detection therefore
needs the teacher LLM again (one forward over the re-tokenized text, logits
and entropy shifted so position t describes the distribution that generated
token t — upstream src/main.py's `[0] + x[:-1]` convention); removing that
requirement is precisely what the WPN buys in the full method.

Port deltas (same conventions as wm/codewm/codewm.py; none change the
algorithm):
  - the pair-choice RNG is a CPU torch.Generator on both sides (upstream
    seeds a generator on the model device; CPU keeps greenlists
    device-independent, and both sides of every run go through this port).
  - detection logits/entropy are computed in float32 (upstream sorts the
    raw bf16 forward logits while HF generate hands the processor float32
    scores; float32 on both sides minimizes near-tie rank mismatch).
  - detection dicts carry this repo's unified keys (z_score /
    is_watermarked) alongside the upstream ones (prediction, confidence).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import scipy.stats
import torch
from transformers import LogitsProcessor


@dataclass
class AcwSConfig:
    vocab_size: int
    gamma: float = 0.5  # pair selection fixes the green fraction at 1/2
    delta: float = 2.0
    hash_key: int = 15485863
    entropy_threshold: float = 1.2  # upstream run.sh / main.py default
    z_threshold: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


def _pair_greenlist(scores_1d: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Upstream SweetCodeMixin._get_greenlist_ids: sort logits descending,
    pair adjacent ranks, seeded coin picks one member per pair (the odd
    last rank joins on coin=1)."""
    sorted_indices = torch.sort(scores_1d, descending=True).indices
    seq_len = sorted_indices.shape[0]
    k = (seq_len + 1) // 2
    random_choices = torch.randint(0, 2, (k,), generator=rng)
    base_indices = torch.arange(0, seq_len - 1, 2)
    selected = base_indices + random_choices[: len(base_indices)]
    if seq_len % 2 == 1 and random_choices[-1].item():
        selected = torch.cat([selected, torch.tensor([seq_len - 1])])
    return sorted_indices[selected.to(sorted_indices.device)]


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """Upstream's Shannon entropy over the last dim."""
    return -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(-1)


class AcwSLogitsProcessor(LogitsProcessor):
    def __init__(self, config: AcwSConfig):
        self.config = config
        self.rng = torch.Generator()  # CPU (see port deltas)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx in range(input_ids.shape[0]):
            self.rng.manual_seed(self.config.hash_key * int(input_ids[b_idx, -1]))
            greenlist_ids = _pair_greenlist(scores[b_idx].float(), self.rng)
            green_tokens_mask[b_idx][greenlist_ids.to(scores.device)] = True

        ent = _entropy(torch.softmax(scores.float(), dim=-1))
        entropy_mask = (ent > self.config.entropy_threshold).view(-1, 1)
        green_tokens_mask = green_tokens_mask * entropy_mask.to(green_tokens_mask.device)
        scores[green_tokens_mask] = scores[green_tokens_mask] + self.config.delta
        return scores


class AcwSDetector:
    """Upstream SweetCodeDetector._score_sequence, with the main.py logits/
    entropy plumbing folded in: detect() takes the full re-tokenized text
    plus the prompt length and runs the teacher forward itself."""

    def __init__(self, config: AcwSConfig, model):
        self.config = config
        self.model = model
        self.rng = torch.Generator()  # CPU (see port deltas)

    @torch.no_grad()
    def detect(self, input_ids: torch.Tensor, prefix_len: int) -> dict:
        device = next(self.model.parameters()).device
        ids = input_ids.to(device)
        num_tokens_generated = len(ids) - prefix_len
        if num_tokens_generated < 1:
            return {"invalid": True, "z_score": None, "is_watermarked": False}

        logits = self.model(ids.unsqueeze(0), return_dict=True).logits[0].float()
        ent_raw = _entropy(torch.softmax(logits, dim=-1))
        # upstream main.py shift: entropy[t] / scores[t] describe the
        # distribution that generated token t
        entropy = torch.cat([ent_raw.new_zeros(1), ent_raw[:-1]])
        scores = torch.cat([torch.zeros_like(logits[:1]), logits[:-1]])

        ids_cpu = ids.cpu()
        green_token_count = 0
        num_tokens_scored = 0
        for idx in range(prefix_len, len(ids)):
            if float(entropy[idx]) <= self.config.entropy_threshold:
                continue
            num_tokens_scored += 1
            self.rng.manual_seed(self.config.hash_key * int(ids_cpu[idx - 1]))
            greenlist_ids = _pair_greenlist(scores[idx], self.rng)
            green_token_count += bool((greenlist_ids == ids[idx]).any())

        out = {
            "num_tokens_generated": num_tokens_generated,
            "num_tokens_scored": num_tokens_scored,
            "num_green_tokens": green_token_count,
            "watermarking_fraction": num_tokens_scored / num_tokens_generated,
        }
        if num_tokens_scored == 0:
            out.update({"green_fraction": 0.0, "z_score": 0.0, "p_value": 1.0})
        else:
            g = self.config.gamma
            z = (green_token_count - g * num_tokens_scored) / sqrt(
                num_tokens_scored * g * (1 - g)
            )
            out.update({
                "green_fraction": green_token_count / num_tokens_scored,
                "z_score": z,
                "p_value": float(scipy.stats.norm.sf(z)),
            })
        prediction = bool(out["z_score"] > self.config.z_threshold)
        out["prediction"] = prediction
        out["is_watermarked"] = prediction
        if prediction:
            out["confidence"] = 1 - out["p_value"]
        return out
