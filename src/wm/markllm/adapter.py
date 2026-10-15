"""Shared plumbing for the MarkLLM-package reproduction.

Everything watermark-algorithmic here delegates to the cloned MarkLLM repo
(llm_watermark/MarkLLM): SBERT+LSH via `SemStampUtils.SBERTLSHModel` (nearpy),
valid-region masks via `SemStampUtils.rng` (their exact seeding), detection
chain per `SemStamp.detect_watermark`. The only local additions are (a) a
cached lsh_model — MarkLLM re-loads the SBERT embedder inside every
generate/detect call — and (b) a pluggable segmenter, because MarkLLM's
`sent_tokenize` is degenerate on code (see pilot README).

NOTE on `import utils`: MarkLLM has a top-level `utils` package which clashes
with the pilot's src/utils.py — and it is a NAMESPACE package (no
__init__.py), so a regular utils.py module anywhere on sys.path outranks it.
src/ must therefore never be on sys.path here: pilot helpers are loaded via
importlib under the name `pilot_utils` (load_pilot_utils), and the pilot
`dataloader` package is registered in sys.modules explicitly at import time,
so runners can still `import dataloader`.
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

# Default: MarkLLM cloned next to this repository's checkout; override with the
# MARKLLM_ROOT environment variable.
MARKLLM_ROOT = Path(os.environ.get("MARKLLM_ROOT", Path(__file__).resolve().parents[4] / "MarkLLM"))
REPO_SRC = Path(__file__).resolve().parents[2]

if not MARKLLM_ROOT.is_dir():
    raise FileNotFoundError(
        f"MarkLLM repo not found at {MARKLLM_ROOT}. The MarkLLM reproduction "
        "needs a clone of https://github.com/THU-BPM/MarkLLM: either clone it "
        "there or point the MARKLLM_ROOT environment variable at an existing "
        "clone."
    )
if str(MARKLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MARKLLM_ROOT))


def load_pilot_utils():
    spec = importlib.util.spec_from_file_location("pilot_utils", REPO_SRC / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _register_pilot_package(name: str):
    """Make src/<name>/ importable under its own name without putting src/ on
    sys.path (which would shadow MarkLLM's namespace `utils` package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_SRC / name / "__init__.py",
        submodule_search_locations=[str(REPO_SRC / name)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_register_pilot_package("dataloader")


def make_transformers_config(tokenizer=None, vocab_size=0, device="cpu"):
    from utils.transformers_config import TransformersConfig  # MarkLLM's utils

    return TransformersConfig(
        model=None, tokenizer=tokenizer, vocab_size=vocab_size, device=device
    )


def segment(text: str, mode: str) -> list[str]:
    if mode == "sent":  # MarkLLM original
        from nltk.tokenize import sent_tokenize

        return [s for s in sent_tokenize(text) if s.strip()]
    if mode == "line":  # pilot's code-adapted segmenter
        return [l.strip() for l in text.split("\n") if l.strip()]
    raise ValueError(mode)


class MarkLLMSemStamp:
    """MarkLLM SemStamp components with a cached embedder."""

    def __init__(self, device: str = "cpu", **config_overrides):
        import torch

        from watermark.semstamp.semstamp import SemStampConfig, SemStampUtils

        self.torch = torch
        tc = make_transformers_config(device=device)
        self.config = SemStampConfig(
            str(MARKLLM_ROOT / "config" / "SEMSTAMP.json"), tc, **config_overrides
        )
        self.utils = SemStampUtils(self.config)
        # MarkLLM instantiates SBERTLSHModel inside every call; cache one.
        self.lsh = SemStampUtils.SBERTLSHModel(
            lsh_model_path=self.config.path_to_embedder,
            batch_size=32,
            lsh_dim=self.config.dimension_d,
            sbert_type="base",
        )
        self.n_bins = 2 ** self.config.dimension_d
        self.n_accept = int(self.n_bins * self.config.gamma)
        self.normals = torch.tensor(self.lsh.hasher.normals, dtype=torch.float32)

    def hash_texts(self, texts: list[str]) -> list[int]:
        return list(self.lsh.get_hash(texts))

    def embed(self, texts: list[str]):
        return self.lsh.get_embeddings(texts)

    def sigs_and_margins(self, embeds) -> tuple[list[int], list[float]]:
        """LSH signature (their hash_vector path) + min |cos to normals|
        (their margin test in generate_watermarked_text)."""
        import numpy as np
        import torch.nn.functional as F

        hash_strs = [self.lsh.hasher.hash_vector(e)[0] for e in np.asarray(embeds)]
        sigs = [int(s, 2) for s in hash_strs]
        e = self.torch.tensor(np.asarray(embeds), dtype=self.torch.float32)
        sims = F.cosine_similarity(
            e.unsqueeze(1).expand(-1, self.normals.size(0), -1).reshape(-1, e.size(1)),
            self.normals.unsqueeze(0).expand(e.size(0), -1, -1).reshape(-1, e.size(1)),
        ).view(e.size(0), self.normals.size(0))
        margins = sims.abs().min(dim=1).values.tolist()
        return sigs, margins

    def accept_mask(self, lsh_seed: int) -> set[int]:
        """Exact MarkLLM derivation (semstamp.py generate/detect)."""
        self.utils.rng.manual_seed(self.config.prime_P * lsh_seed)
        perm = self.torch.randperm(
            self.n_bins, device=self.config.device, generator=self.utils.rng
        )
        return set(perm[: self.n_accept].tolist())

    def detect(self, text: str, segmenter: str) -> dict:
        """MarkLLM SemStamp.detect_watermark chain, with pluggable segmenter
        and a cached embedder. segmenter='sent' is behaviour-identical to the
        package (modulo batched embedding)."""
        sentences = segment(text, segmenter)
        n_sent = len(sentences)
        if n_sent < 2:
            return {
                "z_score": None,
                "n_segments": n_sent,
                "n_tested": 0,
                "valid_fraction": None,
                "is_watermarked": False,
            }
        hashes = self.hash_texts(sentences)
        lsh_seed = hashes[0]
        n_watermark = 0
        for cand in hashes[1:]:
            if cand in self.accept_mask(lsh_seed):
                n_watermark += 1
            lsh_seed = cand
        n_test = n_sent - 1
        g = self.config.gamma
        z = (n_watermark - g * n_test) / math.sqrt(n_test * g * (1 - g))
        return {
            "z_score": z,
            "n_segments": n_sent,
            "n_tested": n_test,
            "n_valid": n_watermark,
            "valid_fraction": n_watermark / n_test,
            "is_watermarked": bool(z > self.config.threshold),
        }
