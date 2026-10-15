# Local research documents

| Document | Scope |
|---|---|
| [Baseline report](baseline_benchmarks.pdf) ([LaTeX](baseline_benchmarks.tex)) | Five-method measured scope, conventions, complete primary table, original-paper CodeWM table, limitations |
| [Detailed results](../out/README.md) | Full measured values, SeqMark search and threshold classification, figures, superseded runs |

Build the current PDFs locally:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error baseline_benchmarks.tex
pdflatex -interaction=nonstopmode -halt-on-error baseline_benchmarks.tex
```

`INTERFACE.md` defines the current results interface. No external Overleaf
project is required or configured.

The first-round report is packaged in
[the first-round archive](../results/archives/round1-01.tar.gz). Restore it
with `python3 src/result_archive.py restore --group round1` from the repository
root; it appears at `archive/out_round1/README.md`.

The evidence-based SemStamp control-value correction is recorded in
[report_corrections.json](../results/report_corrections.json).
