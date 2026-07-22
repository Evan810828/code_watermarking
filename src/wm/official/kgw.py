"""Official KGW from jwkirchenbauer/lm-watermarking (extended processor:
WatermarkLogitsProcessor for generation, WatermarkDetector for detection on
the decoded completion text) — not MarkLLM and not the pilot
re-implementation."""

import torch

from . import add_repo_paths

add_repo_paths()
from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor  # noqa: E402


def _sanitize(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        else:
            out[k] = float(v)
    return out


class OfficialKGW:
    """Logits processor + text detector with the pilot's unified detection
    keys (z_score / is_watermarked) added alongside the official ones."""

    def __init__(self, tokenizer, device, gamma, delta, seeding_scheme,
                 z_threshold, ignore_repeated_ngrams):
        vocab = list(tokenizer.get_vocab().values())
        self.logits_processor = WatermarkLogitsProcessor(
            vocab=vocab, gamma=gamma, delta=delta, seeding_scheme=seeding_scheme
        )
        self.detector = WatermarkDetector(
            vocab=vocab,
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            device=device,
            tokenizer=tokenizer,
            z_threshold=z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=ignore_repeated_ngrams,
        )
        self.meta = {
            "algorithm": "KGW",
            "implementation": "official jwkirchenbauer/lm-watermarking (extended_watermark_processor)",
            "gamma": gamma,
            "delta": delta,
            "seeding_scheme": seeding_scheme,
            "z_threshold": z_threshold,
            "ignore_repeated_ngrams": ignore_repeated_ngrams,
            "normalizers": [],
        }

    def detect(self, completion: str, full_ids=None, prompt_len=None) -> dict:
        try:
            det = _sanitize(
                self.detector.detect(
                    text=completion,
                    return_green_token_mask=False,
                    return_z_at_T=False,
                    convert_to_float=True,
                )
            )
        except ValueError as e:  # too short to score
            det = {"error": str(e), "prediction": False, "z_score": None}
        det["is_watermarked"] = bool(det.get("prediction"))
        return det
