"""SWEET watermark (Lee et al. 2024, "Who Wrote this Code? Watermarking for
Code Generation", https://github.com/hongcheki/sweet-watermark).

Port of the repo's core (sweet.py + watermark.py WatermarkBase, Apache-2.0)
— not the whole repo. SWEET is KGW with selective entropy gating, built for
exactly the failure mode this pilot found on code (most code tokens are
near-deterministic, so uniform green-list biasing either does nothing or
destroys the code):

  - seeding is KGW "simple_1" (= lefthash with 1-token prefix): the rng is
    seeded with hash_key * previous_token and gamma * vocab_size ids are
    drawn as the green list;
  - generation applies the delta bias ONLY at positions where the model's
    predictive entropy exceeds entropy_threshold;
  - detection z-tests ONLY tokens whose entropy exceeds the threshold, so
    the forced low-entropy tokens no longer dilute the statistic. Entropy is
    recomputed with the same model over prompt + completion and shifted
    right by one (upstream lm_eval/evaluator.py: `entropy = [0] +
    entropy[:-1]`) so entropy[i] is the entropy of the distribution that
    produced token i.

Port deltas (documented, none change the algorithm):
  - config vocab_size instead of upstream's len(tokenizer vocab list);
    generation and detection use the same value, which is all that matters.
  - detection skips the green-list draw for low-entropy tokens instead of
    drawing and discarding (the rng is re-seeded per position, so this is
    an efficiency-only change with identical output).
  - the detection dict carries this repo's unified keys (z_score /
    is_watermarked) alongside the upstream ones (prediction, confidence),
    like wm/MarkLLM_official/kgw.
  - upstream's z=-100 "no scored tokens" sentinel is kept, with
    num_tokens_scored=0 marking the case.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from scipy import stats
from transformers import LogitsProcessor


@dataclass
class SweetConfig:
    vocab_size: int
    gamma: float = 0.5
    delta: float = 8.0
    hash_key: int = 15485863
    entropy_threshold: float = 0.5
    z_threshold: float = 4.0
    select_green_tokens: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(logits) over the last dim (upstream
    lm_eval/utils.py calculate_entropy)."""
    probs = torch.softmax(logits, dim=-1)
    return -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)


class SweetBase:
    """WatermarkBase port: simple_1 seeding + green-list draw (CPU rng, as
    upstream)."""

    def __init__(self, config: SweetConfig):
        self.config = config
        self.rng = torch.Generator()

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        assert input_ids.shape[-1] >= 1, "simple_1 seeding needs a 1-token prefix"
        self.rng.manual_seed(self.config.hash_key * int(input_ids[-1]))
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, generator=self.rng)
        if self.config.select_green_tokens:
            return vocab_permutation[:greenlist_size]
        return vocab_permutation[self.config.vocab_size - greenlist_size:]


class SweetLogitsProcessor(SweetBase, LogitsProcessor):
    """SweetLogitsProcessor port: KGW green-list bias gated on the entropy of
    the raw scores (runs before HF's temperature/top-p warpers, as upstream)."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx in range(input_ids.shape[0]):
            green_tokens_mask[b_idx][self._get_greenlist_ids(input_ids[b_idx])] = True
        ent = predictive_entropy(scores)
        entropy_mask = (ent > self.config.entropy_threshold).view(-1, 1)
        green_tokens_mask = green_tokens_mask * entropy_mask
        scores[green_tokens_mask] = scores[green_tokens_mask] + self.config.delta
        return scores


class SweetDetector(SweetBase):
    """SweetDetector._score_sequence port. Needs the generating model to
    recompute per-token entropy, so detection runs on token ids with their
    true (unpadded) left context."""

    def __init__(self, config: SweetConfig, model):
        super().__init__(config)
        self.model = model

    def compute_entropy(self, full_ids: torch.LongTensor) -> list[float]:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(full_ids.unsqueeze(0).to(device), return_dict=True).logits[0]
        ent = predictive_entropy(logits).float().cpu().tolist()
        # shift right so ent[i] is the entropy of the distribution that
        # produced token i; position 0 gets a dummy (never scored)
        return [0.0] + ent[:-1]

    def detect(self, full_ids: torch.LongTensor, prefix_len: int) -> dict:
        full_ids = full_ids.cpu()
        prefix_len = max(1, prefix_len)
        num_tokens_generated = len(full_ids) - prefix_len
        if num_tokens_generated < 1:
            return {"invalid": True, "z_score": None, "is_watermarked": False}

        entropy = self.compute_entropy(full_ids)
        green_token_count = 0
        num_tokens_scored = 0
        for idx in range(prefix_len, len(full_ids)):
            if entropy[idx] > self.config.entropy_threshold:
                num_tokens_scored += 1
                if full_ids[idx] in self._get_greenlist_ids(full_ids[:idx]):
                    green_token_count += 1

        if num_tokens_scored < 1:
            # upstream: no high-entropy token -> regard as human-written
            return {
                "num_tokens_generated": num_tokens_generated,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "watermarking_fraction": 0.0,
                "green_fraction": 0.0,
                "z_score": -100.0,
                "p_value": 1.0,
                "prediction": False,
                "is_watermarked": False,
            }

        g = self.config.gamma
        z = (green_token_count - g * num_tokens_scored) / math.sqrt(
            num_tokens_scored * g * (1 - g)
        )
        p_value = float(stats.norm.sf(z))
        prediction = bool(z > self.config.z_threshold)
        out = {
            "num_tokens_generated": num_tokens_generated,
            "num_tokens_scored": num_tokens_scored,
            "num_green_tokens": green_token_count,
            "watermarking_fraction": num_tokens_scored / num_tokens_generated,
            "green_fraction": green_token_count / num_tokens_scored,
            "z_score": z,
            "p_value": p_value,
            "prediction": prediction,
            "is_watermarked": prediction,
        }
        if prediction:
            out["confidence"] = 1 - p_value
        return out
