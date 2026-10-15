# Watermarking baselines for code generation

A measurement and exploration repository for KGW, SemStamp, SWEET, CodeWM,
and SeqMark on HumanEval, MBPP, and LiveCodeBench. It contains method
implementations, dataset integrations, existing benchmark results, and the
original evidence needed to inspect those results.

[Full results](out/README.md) · [Evidence archives](results/README.md) ·
[Experiment index](results/experiments.csv) ·
[Local research report](docs/baseline_benchmarks.pdf) ·
[Implementation](IMPLEMENTATION.md)

## Methods and measured coverage

| Method | Implementations | HumanEval | MBPP test | LiveCodeBench v1 |
|---|---|---|---|---|
| KGW | Local pilot, original implementation, historical MarkLLM/vLLM | Complete | Complete | Complete |
| SemStamp | Local pilot, original implementation, historical MarkLLM/vLLM | Complete | Complete | Complete |
| SWEET | Local port | Complete | Complete | Complete |
| CodeWM | Local port and original-paper completion protocol | Full and capped variants | Full and capped variants | Capped variants |
| SeqMark | Complete-program port with Gemma and SBERT encoders | Complete, both encoders | Complete, both encoders | Cancelled; partial evidence retained |

Full split sizes are 164, 500, and 400 tasks, respectively. Coverage describes
available measurements, not all possible method/implementation combinations.
The report retains parameter variants, the first experimental round,
original-paper reproductions, and superseded results separately.

Detection uses extracted Python programs and paired unwatermarked model
outputs. The primary table reports AUROC, true positive rate at 1% and 5%
false positive rate, mean detector scores, scoreability, and pass@1 for both
branches. Tied scores remain together at deterministic thresholds. Read
scoreability and cohort size alongside detection metrics.

SeqMark uses different sampling settings from the other baselines. CodeWM's
original-paper protocol uses completion prompts and a 20-sample pass@1
estimator. These settings are documented in the report and must not be
interpreted as a controlled method-only comparison.

## Inspect existing results without a GPU

```bash
git clone https://github.com/Evan810828/code_watermarking.git
cd code_watermarking
python3 src/result_archive.py verify
python3 src/result_archive.py restore
python3 -m pip install numpy
cd src
python3 utils.py summarize-methods
python3 utils.py summarize-seqmark
```

Archive verification and restoration use only the Python standard library.
The summary commands need Python 3.10 or newer and NumPy (`python3 -m pip
install numpy`); generation uses the Python 3.12 environment below.
Restore retains original file bytes and paths, and refuses to overwrite a
file with different contents. Use `--destination /path/to/empty-directory`
for an independent copy. The archive inventory includes hashes for every
file. Model weights and download caches are excluded.

## Install generation dependencies

```bash
conda create -n watermark python=3.12 -y
conda activate watermark
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

The requirements retain reference-environment version constraints. They are
not a complete environment lockfile. Upstream KGW and SemStamp adapters
require the corresponding source repository next to this repository, or an
environment-variable override:

| Adapter | Upstream repository | Override |
|---|---|---|
| KGW `--impl official` | [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) | `LM_WATERMARKING_DIR` |
| SemStamp `--impl official` | [SemStamp](https://github.com/bohanhou14/SemStamp) | `SEMSTAMP_DIR` |
| MarkLLM/vLLM | [MarkLLM](https://github.com/THU-BPM/MarkLLM) | `MARKLLM_ROOT` |

Historical MarkLLM/vLLM scripts use a separate environment (vLLM 0.18,
transformers 4.57) and run from `src/wm/markllm/`. Keep that environment
separate from the Hugging Face runners. Upstream repositories remain
external dependencies and retain their own licenses.

## Generate and evaluate

Run the method entry points from `src/`. Each runner accepts a dataset;
prompts and benchmark evaluation are defined in `src/dataloader/`.

```bash
python run_kgw.py --dataset humaneval --impl pilot
python run_semstamp.py --dataset mbpp --impl official --max-trials 32
python run_sweet.py --dataset livecodebench
python run_codewm.py --dataset mbpp --code-model /path/to/best_rank.pt

# SeqMark with the selected scientific parameters and a new HF output path.
python run_seqmark.py --dataset humaneval --backend hf \
  --temperature 1.2 --top-p 1.0 --top-k -1 --repetition-penalty 1.05 \
  --center-samples 50 --center-temperature 1.2 --max-trials 50 \
  --lsh-dim 5 --gamma 0.1 --detector-center generation_cache \
  --output ../out/seqmark/humaneval/seqmark_humaneval_port_hf_qwen3.5-4b.json

python eval_pass1.py --input ../out/kgw/humaneval/<run>.json --tag kgw
```

The SeqMark command uses `google/embeddinggemma-300m` by default. The selected
parameters are recorded in `out/seqmark/sweep/selected_config.json`. The recorded runs used vLLM. Select
`--backend vllm` in the corresponding environment to use that engine.
Changing the backend does not guarantee identical samples. The complete
configuration and source revisions are recorded in the saved manifests.
`--selected-config` validates the original backend and exact local model paths;
use it only when reproducing that recorded environment.

CodeWM requires a trained CodeModel. `train_codewm.py gen`, `build`, and
`train` implement the chat-protocol training pipeline. The preserved
original-paper inference protocol is `run_codewm_repro.py`; it additionally
requires the historical per-model checkpoint and has different evaluation
semantics. See [implementation details](IMPLEMENTATION.md).

HumanEval and MBPP task lists include their evaluation material. LiveCodeBench
execution additionally needs the release-v1 per-task tests, which are not
included in Git. Prepare them with:

```bash
python -m dataloader.livecodebench --jsonl /path/to/test.jsonl
```

This writes `data/lcb/tests/`, or `$LCB_SCRATCH/tests/` when configured.
The stored protocol evaluates at most 20 tests per task and skips test inputs
larger than 10 MB during preparation. `eval_pass1.py` executes generated code;
use an isolated execution environment.

## Repository layout and documents

| Path | Contents |
|---|---|
| `src/` | Baseline methods, parameterized runners, evaluation, result utilities |
| `data/` | Slim task lists for the three benchmarks |
| `out/README.md` | Detailed measured results and historical comparisons |
| `results/` | Compressed original evidence, inventory, restoration instructions |
| `docs/` | Local LaTeX/PDF baseline report and document index |
| `PROJECT.md` | Research scope |
| `INTERFACE.md` | Evaluation and preservation contract |
| `IMPLEMENTATION.md` | Code boundaries, commands, dependencies, provenance limitations |

No new experiments were run for the repository reorganization. Documentation
is maintained locally; no external Overleaf project is configured.
