# Implementation

## Repository map

| Location | Responsibility |
|---|---|
| `src/wm/` | KGW, SemStamp, SWEET, CodeWM, SeqMark, and upstream adapters |
| `src/dataloader/` | Task loading, prompts, extraction context, and execution evaluation |
| `src/run_*.py` | Dataset-parameterized method runners |
| `src/train_codewm.py` | Chat-protocol CodeModel corpus generation, distillation data, training |
| `src/eval_pass1.py` | Functional evaluation of saved generations |
| `src/utils.py` | Shared generation helpers, detection metrics, merging, and summaries |
| `src/result_archive.py` | Standard-library archive verification and restoration |
| `src/similarity_study.py` | Existing semantic-diversity exploration |
| `src/plot_tradeoff.py` | Historical two-dataset, four-method plot |
| `data/` | Three slim benchmark task lists |
| `out/README.md` | Full results, settings, limitations, and superseded values |
| `results/` | Immutable compressed evidence, file inventory, and experiment index |
| `docs/` | Local baseline report and document index |
| `out/`, `logs/`, `archive/` | Restored evidence and new generated artifacts |
| `runs/<run-id>/` | Prospective execution records |

The inference boundary is dataset integration, watermark implementation,
and shared generation helpers. Dataset prompts belong in `dataloader`;
chat templates belong in `utils`. Do not add per-dataset runners.
The Python `ast` module in `extract_program` validates Python syntax and
remains necessary for the existing extraction protocol.

## Environments and upstream sources

Use Python 3.12 and `requirements.txt` for the Hugging Face runners. The
requirements retain the recorded reference-environment minimum versions;
they are not a complete lockfile. MarkLLM/vLLM reproductions use a separate
historical environment (vLLM 0.18, transformers 4.57). Do not install that
transformers version over the Hugging Face environment. Historical environment
identities, where available, remain in the archived manifests.

Upstream adapters resolve sibling clones using `LM_WATERMARKING_DIR`,
`SEMSTAMP_DIR`, or `MARKLLM_ROOT`. MarkLLM has its own `utils` namespace:
its runners load this repository's helpers through `adapter.load_pilot_utils()`.
Run those entry points from `src/wm/markllm/`.

The CodeWM chat runner uses `--code-model` or `CODEWM_CKPT`; training artifacts
use `CODEWM_WORK_DIR` or `out/codewm_train/`. The restored original-paper
runner is `src/run_codewm_repro.py`, with its protocol in
`src/wm/codewm/repro_protocol.py`. Its completion prompts and 20-sample
estimator differ from the chat protocol. Historical upstream-style training
corpora and checkpoints are external inputs; the present trainer generates
only the chat-protocol corpus.

## Saved evidence and reproducibility

From the repository root:

```bash
python3 src/result_archive.py verify
python3 src/result_archive.py restore
cd src
python utils.py summarize-methods
python utils.py summarize-seqmark
```

Restore verifies every selected archive and refuses to replace an existing
file with different bytes. `--destination /path/to/empty-directory` creates an
independent evidence tree. The inventory is historical execution evidence,
not a reconstructed run ledger: missing commands, hardware identities, or
revisions remain unavailable. Archived source snapshots are immutable and
may use the paths of the original workspace.

`utils.py summarize-methods` recomputes detection from saved scores; it does
not rerun models or regenerate the entire prose report. Original-method
pass@1 values remain in evaluator files. SeqMark summaries verify input and
verdict checksums before combining detection and quality.

LiveCodeBench execution needs separately prepared tests under
`data/lcb/tests/` or `$LCB_SCRATCH/tests/`. The stored protocol caps tests at
20 per task and skips inputs larger than 10 MB during preparation. Existing
scores can be inspected without downloading those tests. Model weights and
CodeModel checkpoints are not included in the result archives.

## Local documents

Compile `docs/baseline_benchmarks.tex` locally with pdfLaTeX. Historical
results and conventions are recorded in `out/README.md` and the evidence
archives. No external document pairing is configured.
