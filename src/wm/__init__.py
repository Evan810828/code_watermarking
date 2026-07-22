"""Watermarking-method code, one subpackage per implementation family:

    wm.kgw       pilot KGW re-implementation (MarkLLM algorithm, HF logits processor)
    wm.semstamp  pilot SemStamp re-implementation (MarkLLM algorithm)
    wm.official  adapters around the authors' original repos
                 (lm-watermarking for KGW, bohanhou14/SemStamp)
    wm.markllm   adapter + vLLM runners for the THU-BPM/MarkLLM package

Like dataloader/, nothing in this package may import the top-level
src/utils.py (name clash with MarkLLM's `utils` package).
"""
