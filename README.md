# Code Watermarking Pilot Study

Pilot study of LLM watermarking methods — **KGW**, **SemStamp**, **SWEET**,
and **CodeWM** (Guo & Cheng, NeurIPS 2025) — on code generation benchmarks
(**HumanEval**, **MBPP**, **LiveCodeBench**) with Qwen3.5-4B.

## Methods & Datasets

| Method | Paper | Venue | Official code | Implementation here |
|---|---|---|---|---|
| **KGW** | [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) (Kirchenbauer et al.) | ICML 2023 | [jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) | `run_kgw.py`; `wm/kgw/` (pilot) or `wm/MarkLLM_official/` (official) |
| **SemStamp** | [SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation](https://arxiv.org/abs/2310.03991) (Hou et al.) | NAACL 2024 | [bohanhou14/SemStamp](https://github.com/bohanhou14/SemStamp) | `run_semstamp.py`; `wm/semstamp/` (pilot) or `wm/MarkLLM_official/` (official) |
| **SWEET** | [Who Wrote this Code? Watermarking for Code Generation](https://arxiv.org/abs/2305.15060) (Lee et al.) | ACL 2024 | [hongcheki/sweet-watermark](https://github.com/hongcheki/sweet-watermark) | `run_sweet.py`; `wm/sweet/` (core-only port) |
| **CodeWM** | [Practical and Effective Code Watermarking for Large Language Models](https://openreview.net/forum?id=RpE4HeuX69) (Guo & Cheng) | NeurIPS 2025 | [TimeLovercc/code-watermark](https://github.com/TimeLovercc/code-watermark) | `run_codewm.py` + `train_codewm.py`; `wm/codewm/` (core-only port) |

KGW and SemStamp are additionally reproduced through the
[THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM) toolkit with vLLM
inference (`src/wm/markllm/`).

| Dataset | Tasks used | Paper | Source |
|---|---|---|---|
| **HumanEval** | 164 (full set) | [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) | [openai/human-eval](https://github.com/openai/human-eval) |
| **MBPP** | 500 (test split) | [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| **LiveCodeBench** | 400 (release v1, code_generation test split) | [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974) | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) |

## Setup

```bash
git clone https://github.com/Evan810828/code_watermarking.git
cd code_watermarking
conda create -n watermark python=3.12 -y
conda activate watermark
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```



### Optional: official / MarkLLM reproductions

`--impl official` (via `src/wm/MarkLLM_official/`) and `src/wm/markllm/`
import from the authors' original repos. Clone whichever you need **next to
this repo** (or point the environment variable at an existing clone):

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

# semantic-diversity study (any subset of datasets; default: humaneval mbpp)
python similarity_study.py --datasets humaneval mbpp livecodebench

# shard merging / summaries (merge-semstamp: humaneval|mbpp; LCB has its own)
python utils.py merge-semstamp --dataset humaneval --num-shards 4
python utils.py merge-semstamp-lcb --num-shards 6 --impl official
python utils.py summarize-lcb
```



CodeWM needs its trained CodeModel first
(`python train_codewm.py gen|build|train`); artifacts default to
`out/codewm_train/` (relocate with `$CODEWM_WORK_DIR`, or point
`$CODEWM_CKPT` / `--code-model` at an existing checkpoint).

## Conventions

- **No per-dataset scripts.** Every run is composed of a dataloader module
  (`src/dataloader/<dataset>.py`), a watermark implementation (`src/wm/`),
  and the dataset-agnostic runners. To add a benchmark, write one dataloader
  module and register it in `dataloader/__init__.py`.
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
