"""Core-only port of ../code-watermark (Guo & Cheng, NeurIPS 2025): trained
pairwise-rank greenlists with a learned entropy switch; detection needs only
the small CodeModel, never the teacher LLM. See codewm.py for the algorithm
and the documented port deltas; training lives in src/train_codewm.py."""

from .codewm import CodeDetector, CodeLogitsProcessor, CodeWmConfig
from .model import CodeModel

__all__ = ["CodeDetector", "CodeLogitsProcessor", "CodeWmConfig", "CodeModel"]
