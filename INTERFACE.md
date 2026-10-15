# Benchmark and preservation contract

## Repository acceptance

- Retain the five baseline method implementations and three dataset integrations.
- Preserve existing baseline generation, detection, execution evaluation,
  parameter-search, diagnostic, and historical reproduction evidence byte for byte.
- Remove the structural-carrier capacity and augmentation research code,
  configurations, and active documentation.
- Provide readable reports, an experiment index, checksums, and restoration
  commands that work in a fresh clone without model downloads.
- Keep documentation local. No external Overleaf configuration is required.

## Scientific outputs

The existing primary table in [out/README.md](out/README.md) reports method,
implementation, dataset, n, scored, unscoreable, AUROC, TPR at 1% and 5% FPR,
mean z-score for both branches, and pass@1 for both branches. It is retained without new measurements. Reporting corrections are recorded
in `results/report_corrections.json` with the original display and source hashes. Sources are the corresponding method/dataset JSON
files indexed in [results/experiments.csv](results/experiments.csv).

Detection uses extracted programs and paired unwatermarked model outputs.
Scoreable classes have separate denominators. Deterministic thresholds keep
ties together; fixed-FPR TPR uses the maximum attainable TPR without
interpolation. Extraction failures and detector-null scores remain distinct.
Functional evaluation retains the benchmark extraction and full task cohort.
Full benchmark task counts are HumanEval 164, MBPP test 500, and LiveCodeBench
release v1 400; capped and incomplete runs remain explicitly labelled.

SeqMark additionally retains its training-selected threshold classification
table, MBPP training-search table, and existing ROC and score-distribution
figures. These use the complete HumanEval/MBPP runs and their recorded
provenance under `out/seqmark/runs/`. The complete program is one sequence.
No complete SeqMark LiveCodeBench result is claimed. Different sampling
parameters prevent treating SeqMark as a controlled method-only comparison
with the other methods.

The original CodeWM completion-prompt protocol uses a 20-sample pass@1
estimator and first-valid-sample detection. Its historical human-reference
negative class is kept separately from the paired-negative rescored results.
First-round results use shorter generation budgets and separate conventions.
Neither is pooled with the primary chat-protocol comparison.

The two existing SeqMark figures retain their source data, plotting code,
class denominators, one-seed scope, and captions. The older trade-off figure
is a historical diagnostic with a narrower scope; it is not a complete
five-method comparison. Baseline robustness and prompt-ablation evidence are
retained as historical supplementary measurements.

## Verification

Archive SHA-256 checks, restore round trips, dataset/task identity checks,
Python syntax checks, command-line help, and document compilation are
operational checks. They do not constitute new benchmark runs or evidence
for a new scientific claim. No cross-method winner criterion is introduced.
