# Watermarking baseline measurement

This repository measures existing watermarking methods on code generation.
It compares detection, scoreability, and functional correctness on HumanEval,
MBPP test, and LiveCodeBench release v1. The retained methods are KGW,
SemStamp, SWEET, CodeWM, and SeqMark.

The research question is how these methods behave under explicitly recorded
model, sampling, extraction, and evaluation conventions. Baseline ports,
upstream adapters, original-paper reproductions, parameter searches, and
superseded runs are distinguished in the reports and evidence inventory.
The repository does not develop a new structural watermark.

The measured scope includes capped CodeWM cells and a cancelled SeqMark
LiveCodeBench run. Completing those cells, adding methods, and changing
scientific comparisons require an explicit experimental decision.
