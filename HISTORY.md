# Project history

## Baseline benchmarking repository (2026-10-15)

The repository scope is measurement and exploration of existing watermarking
baselines on HumanEval, MBPP, and LiveCodeBench. The user requested removal
of the structural watermark research and preservation of all baseline
benchmarking results and documentation.

The five baseline methods, original-paper CodeWM reproduction, historical
round, parameter searches, partial runs, and existing per-task evidence are
retained. Results are packaged with file-level SHA-256 identities and a
restoration tool. Workflow documents describe the baseline-only scope;
LaTeX documents remain local. No new benchmark execution accompanies this
reorganization.

The result audit corrected the two SemStamp LiveCodeBench control pass@1
displays from 44.3% to 43.8%, based on exact control-text matching and saved
execution verdicts (175/400). The correction record retains the old display
and source hashes; original execution files and the historical report remain
unchanged.
