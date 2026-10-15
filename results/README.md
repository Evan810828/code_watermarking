# Preserved benchmark evidence

[Full results and protocol](../out/README.md) | [Experiment index](experiments.csv) | [File inventory](inventory.json)

The report retains all published local baseline tables, including superseded
values. The inventory covers raw generation, detection, execution verdicts,
searches, run manifests, source snapshots, diagnostics, historical outputs,
and available baseline logs. These are existing measurements, not new runs.

## Restore and verify

```bash
python3 src/result_archive.py verify
python3 src/result_archive.py restore
# Optional: restore one method into an independent evidence directory
python3 src/result_archive.py restore --group seqmark --destination /path/to/empty-directory
```

Commands use only the Python standard library and run from any working
directory. Archives retain repository-relative paths. Restore validates the
archive and every file, retains identical existing files, and rejects
conflicting files before writing. Restoring into this repository makes the
existing `src/utils.py` summary commands work without model downloads.

The inventory records SHA-256 hashes of both archive bytes and each original
file. The experiment index lists generation artifacts with task coverage;
shards and superseded variants are separate rows. The full inventory also
includes artifacts outside that index, including every per-task SeqMark file.

## Archive groups

| Group | Files | Compressed MiB | Contents |
|---|---:|---:|---|
| kgw | 24 | 3.71 | Pilot and official runs on three datasets |
| semstamp | 40 | 5.03 | Pilot and official runs, shards and detection controls |
| sweet | 12 | 2.18 | Three-dataset baseline generation and evaluation |
| codewm | 56 | 13.34 | Chat-protocol variants and original-paper reproduction, including capped cells |
| seqmark | 3215 | 89.53 | Both encoders, complete and partial runs, search, provenance, diagnostics and figures |
| round1 | 65 | 4.06 | First-round KGW/SemStamp, MarkLLM/vLLM, semantic-diversity results and original README |
| robustness | 16 | 2.88 | Historical CodeWM base-checkpoint robustness evidence |
| prompt_and_figures | 6 | 0.27 | Comment-prompt exploration and historical trade-off figures |
| logs | 113 | 0.24 | Available baseline and SeqMark execution logs |

## Coverage and interpretation

The clean comparison covers HumanEval (164 tasks), MBPP test (500), and
LiveCodeBench release v1 (400). KGW, SemStamp, and SWEET have complete cells
on all three datasets. CodeWM includes both full and capped cells. SeqMark
has complete HumanEval and MBPP evaluations for Gemma and SBERT; its
cancelled LiveCodeBench outputs are retained as partial evidence.

First-round generation budgets, legacy detection scopes, original-paper
completion prompts, and SeqMark sampling differ from the main comparison.
Do not pool these rows. `coverage=full` in the index means full task-ID
coverage, not approval for inclusion in the primary table or proof that
functional evaluation is complete. Use the report and recorded metadata.

Historical provenance is preserved as recorded. Absolute paths in manifests
refer to the original workspace and do not make those external resources
part of this repository. Missing execution commands or hardware information
are unavailable. Downloaded model caches, external CodeModel checkpoints,
and large LiveCodeBench test payloads are not benchmark result files and
are not packaged.

The historical robustness measurements remain available for inspection.
Their attack-generation code is outside the active baseline repository.
Restore the `robustness` archive group to inspect the original JSON summaries
and task-level evidence under `out/attacks/`.
