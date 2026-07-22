# Code Watermarking Pilot Study

Pilot study of LLM watermarking methods — **KGW**, **SemStamp**, **SWEET**,
and **CodeWM** (Guo & Cheng, NeurIPS 2025) — on code generation benchmarks
(**HumanEval**, **MBPP**, **LiveCodeBench**) with Qwen3.5-4B.

## Layout

```
data/                     slim benchmark task lists (tracked)
src/
  run_kgw.py              KGW inference        (--impl pilot | official)
  run_semstamp.py         SemStamp inference   (--impl pilot | official)
  run_sweet.py            SWEET inference
  run_codewm.py           CodeWM inference (needs a trained checkpoint)
  train_codewm.py         CodeWM CodeModel training pipeline
  eval_pass1.py           pass@1 evaluation of any inference output
  similarity_study.py     semantic-diversity study
  utils.py                shared helpers + merge/summary CLI
  dataloader/             one module per benchmark (fixed interface)
  wm/                     watermark implementations
    kgw/  semstamp/  sweet/  codewm/    self-contained re-implementations
    official/             adapters around the authors' original repos
    markllm/              MarkLLM-package + vLLM reproduction
  scripts/                slurm shard-launch helpers
out/                      experiment outputs (local, not tracked)
```

Design rule: **no per-dataset scripts.** Every run is composed of a
dataloader module (`src/dataloader/<dataset>.py`), a watermark implementation
(`src/wm/`), and the dataset-agnostic runners. To add a benchmark, write one
dataloader module and register it in `dataloader/__init__.py`.

## Setup

```bash
git clone git@github.com:Evan810828/code_watermarking.git
cd code_watermarking
conda create -n watermark python=3.12 -y
conda activate watermark
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

The self-contained implementations (`--impl pilot`, `run_sweet.py`,
`run_codewm.py`) need nothing else. Models are pulled from the Hugging Face
hub (default `--model Qwen/Qwen3.5-4B`).

### Optional: official / MarkLLM reproductions

`--impl official` and `src/wm/markllm/` import from the authors' original
repos. Clone whichever you need **next to this repo** (or point the
environment variable at an existing clone):

| Needed for | Clone | Env var override |
|---|---|---|
| `run_kgw.py --impl official` | [jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) | `LM_WATERMARKING_DIR` |
| `run_semstamp.py --impl official` | [bohanhou14/SemStamp](https://github.com/bohanhou14/SemStamp) | `SEMSTAMP_DIR` |
| `src/wm/markllm/` scripts | [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM) | `MARKLLM_ROOT` |

The MarkLLM+vLLM scripts run in a separate env (`vllm==0.18`,
`transformers==4.57`), from their own directory (`src/wm/markllm/`).

### Data

`data/` ships the slim task lists for all three benchmarks — HumanEval and
MBPP work out of the box. LiveCodeBench pass@1 additionally needs per-task
test files (~4GB, not in the repo): download the LiveCodeBench
`code_generation` release v1 `test.jsonl` and run the one-time prep

```bash
cd src
python -m dataloader.livecodebench --jsonl /path/to/test.jsonl
```

which writes them to `data/lcb/tests/` (override the location with
`$LCB_SCRATCH`, e.g. a scratch filesystem).

## Running

Everything runs from `src/`:

```bash
# inference: any dataset x any implementation; resumable, checkpointed
python run_kgw.py --dataset humaneval --impl pilot
python run_semstamp.py --dataset livecodebench --impl official --max-trials 32
python run_sweet.py --dataset mbpp

# smoke test any runner
python run_kgw.py --dataset humaneval --impl pilot --limit 4 --batch-size 2

# pass@1 (dataset inferred from the file's meta; executes generated code!)
python eval_pass1.py --input ../out/kgw/humaneval/<file>.json --tag kgw

# semantic-diversity study
python similarity_study.py --datasets livecodebench

# shard merging / summaries
python utils.py merge-semstamp --dataset humaneval --num-shards 4
python utils.py summarize-lcb
```

Long runs shard across GPUs with `--shard/--num-shards`;
`scripts/launch_semstamp_shards.sh <JOBID> <NUM_GPUS>` launches shards inside
an existing slurm allocation (set `$PYTHON` to your interpreter if not using
the `watermark` conda env).

CodeWM needs its trained CodeModel first
(`python train_codewm.py gen|build|train`); artifacts default to
`out/codewm_train/` (relocate with `$CODEWM_WORK_DIR`, or point
`$CODEWM_CKPT` / `--code-model` at an existing checkpoint).

## Conventions

- Sampling defaults (temperature 0.7, top_p 0.8, top_k 20, seed 42,
  `enable_thinking=False`) come from `utils.add_common_args`.
- `--max-new-tokens` defaults to each dataloader's `MAX_NEW_TOKENS`
  (2000 everywhere).
- System prompts are dataset-level (`SYSTEM_PROMPT` in the dataloader) and
  all append the universal no-code-comments rule.
- SemStamp uses the line segmenter (`--segmenter line`) for all experiments.
- KGW/SWEET default to `delta=8` (a delta sweep showed the MarkLLM-default 2
  is undetectable on code).
- Detection dicts always carry `z_score` and `is_watermarked`.
- Runners checkpoint after every batch/task and resume by `task_id` from the
  existing output file. Outputs land in `out/<method>/<dataset>/` via
  `utils.out_dir` — never hardcode output paths.
