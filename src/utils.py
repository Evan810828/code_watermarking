"""Global helpers shared by every runner: paths (out/ is indexed by
watermark method, then dataset, with a subfolder only for reproduction
implementations: out/<method>/<dataset>[/markllm]/), chat-template plumbing,
code extraction, common CLI args, model loading, and the non-experiment CLI
utilities (shard merging, aggregate stats).

Inference follows a fixed three-step pattern (see CLAUDE.md): pick a
dataloader module, pick a watermark implementation, then drive generation
with the global helpers here (chat_text/task_chat_text, extract_code,
add_common_args, load_hf_model). Never bake prompts or system messages into
a runner.

CLI:
    python utils.py merge-semstamp      --dataset humaneval [--num-shards 4]
    python utils.py analyze-semstamp    --dataset humaneval
    python utils.py merge-semstamp-lcb  [--num-shards 6]
    python utils.py summarize-lcb
"""

import argparse
import json
import re
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PILOT_DIR / "data"
OUT_DIR = PILOT_DIR / "out"

SIMILARITY_OUT = OUT_DIR / "similarity"


def out_dir(method: str, dataset: str, impl: str = "pilot") -> Path:
    """out/<method>/<dataset>/. pilot and official runs share the dataset
    folder (disambiguated by the impl segment in the filename); reproduction
    implementations (markllm) get their own subfolder."""
    d = OUT_DIR / method / dataset
    if impl in ("markllm", "markllm_vllm"):
        d = d / "markllm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_kgw_control(dataset: str, tag: str) -> Path:
    """The KGW run whose unwatermarked completions serve as the SemStamp FP
    control. Prefers a new-style pilot file, falls back to the historical
    (pre-reorg) filenames."""
    d = OUT_DIR / "kgw" / dataset
    candidates = [
        f"kgw_{dataset}_pilot_{tag}.json",
        f"kgw_{dataset}_{tag}.json",
        f"kgw_{dataset}_official_{tag}.json",
    ]
    if dataset == "livecodebench":
        candidates.append(f"kgw_lcb_{tag}.json")
    for name in candidates:
        if (d / name).exists():
            return d / name
    return d / candidates[0]


def detect_flag(detection: dict) -> bool:
    """The boolean detection verdict. Reads `is_watermarked`, falling back to
    official KGW's native `prediction`: pre-reorg files (e.g.
    out/kgw/livecodebench/kgw_lcb_*.json) predate the is_watermarked
    convention and only carry prediction."""
    v = detection.get("is_watermarked")
    if v is None:
        v = detection.get("prediction")
    return bool(v)


def model_tag(model_name: str) -> str:
    """'Qwen/Qwen3.5-4B' -> 'qwen3.5-4b' (used in output filenames)."""
    return model_name.split("/")[-1].lower()


# ------------------------------------------------------------------ prompts


def chat_text(tokenizer, user_content: str, system_prompt: str | None = None,
              enable_thinking: bool = False) -> str:
    """The one place a chat template (and optional system message) is applied.
    Dataset system prompts come from the dataloader module's SYSTEM_PROMPT."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def task_chat_text(tokenizer, ds, task: dict) -> str:
    """Chat-template text for one benchmark task (ds: a dataloader module)."""
    return chat_text(tokenizer, ds.build_prompt(task), ds.SYSTEM_PROMPT)


def extract_code(completion: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    # unclosed block (generation hit the token cap before the closing fence)
    m = re.search(r"```(?:python)?\n(.*)$", completion, re.DOTALL)
    return m.group(1).strip() if m else None


# ------------------------------------------------------------------ runners


def add_common_args(ap: argparse.ArgumentParser):
    """Model/sampling/task-slicing args shared by every inference runner
    (Qwen3 non-thinking recommended sampling settings)."""
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=None, help="only run first N tasks")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: the dataset's MAX_NEW_TOKENS")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)


def load_hf_model(model_name: str, device: str, padding_side: str | None = None):
    """Standard HF tokenizer+model load (bf16). Returns (tokenizer, model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"padding_side": padding_side} if padding_side else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


# ---------------------------------------------------------------- non-primary


def merge_semstamp_shards(dataset: str, num_shards: int = 4, tag: str = "qwen3.5-4b") -> Path:
    """Merge sharded pilot run_semstamp outputs into one JSON (task order)."""
    import dataloader

    ds = dataloader.get(dataset)
    results = []
    meta = None
    for i in range(num_shards):
        path = out_dir("semstamp", dataset) / f"semstamp_{dataset}_{tag}.shard{i}of{num_shards}.json"
        with open(path) as f:
            d = json.load(f)
        if meta is None:
            meta = d["meta"]
        results.extend(d["results"])

    task_order = [ds.task_id(t) for t in ds.load_tasks()]
    have = {r["task_id"] for r in results}
    missing = [t for t in task_order if t not in have]
    if missing:
        print(f"WARNING: {len(missing)} tasks missing: {missing[:10]}")
    results.sort(key=lambda r: task_order.index(r["task_id"]))

    meta = dict(meta)
    meta.pop("shard", None)
    meta["num_tasks"] = len(results)
    n_wm = sum(r["watermarked"]["detection"]["is_watermarked"] for r in results)
    ctrl = [r for r in results if "unwatermarked_control" in r]
    n_fp = sum(r["unwatermarked_control"]["detection"]["is_watermarked"] for r in ctrl)
    meta["summary"] = {
        "detect_rate_watermarked": n_wm / len(results),
        "false_positive_rate_unwatermarked": n_fp / len(ctrl) if ctrl else None,
    }
    out_path = out_dir("semstamp", dataset) / f"semstamp_{dataset}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"Merged {len(results)} tasks. TPR={n_wm}/{len(results)}, FP={n_fp}/{len(ctrl)} -> {out_path}")
    return out_path


def analyze_semstamp(dataset: str, tag: str = "qwen3.5-4b") -> None:
    """Aggregate stats for a pilot semstamp run: detection, z distribution,
    rejection-sampling behaviour, candidate semantic diversity."""
    from collections import Counter

    import numpy as np

    with open(out_dir("semstamp", dataset) / f"semstamp_{dataset}_{tag}.json") as f:
        data = json.load(f)
    results = data["results"]

    wm_z = [r["watermarked"]["detection"]["z_score"] for r in results]
    wm_z = [z for z in wm_z if z is not None]
    ctrl = [r["unwatermarked_control"]["detection"] for r in results if "unwatermarked_control" in r]
    ctrl_z = [d["z_score"] for d in ctrl if d["z_score"] is not None]

    segs = [s for r in results for s in r["watermarked"]["segment_log"]]
    status = Counter(s["status"] for s in segs)
    diversity = Counter(s["distinct_candidate_sigs"] for s in segs)
    n_seg = [r["watermarked"]["sampling_stats"]["n_segments"] for r in results]
    trials = [r["watermarked"]["sampling_stats"]["total_trials"] for r in results]
    gen_s = [r["watermarked"].get("gen_seconds") for r in results]
    valid_frac = [
        r["watermarked"]["detection"]["valid_fraction"]
        for r in results
        if r["watermarked"]["detection"]["valid_fraction"] is not None
    ]

    print(f"== SemStamp {dataset} ({len(results)} tasks) ==")
    print(f"summary: {json.dumps(data['meta']['summary'])}")
    print(f"z (wm):   mean={np.mean(wm_z):.2f} median={np.median(wm_z):.2f} "
          f"p95={np.percentile(wm_z, 95):.2f} max={np.max(wm_z):.2f}")
    if ctrl_z:
        print(f"z (ctrl): mean={np.mean(ctrl_z):.2f} median={np.median(ctrl_z):.2f} "
              f"max={np.max(ctrl_z):.2f}  (n={len(ctrl_z)})")
    print(f"valid fraction (wm): mean={np.mean(valid_frac):.3f} (gamma=0.25 => chance)")
    print(f"segments/task: mean={np.mean(n_seg):.1f}  trials/task: mean={np.mean(trials):.0f}")
    print(f"segment status: {dict(status)}")
    print(f"maxed-out rate: {status['maxedout'] / (status['maxedout'] + status['accepted']):.3f}")
    tot = sum(diversity.values())
    print("distinct candidate LSH signatures per segment (of up to 32 trials):")
    for k in sorted(diversity):
        print(f"  {k}: {diversity[k]} ({diversity[k] / tot:.1%})")
    print(f"gen seconds/task: mean={np.mean(gen_s):.1f}")


def merge_semstamp_lcb_shards(num_shards: int = 6, tag: str = "qwen3.5-4b",
                              impl: str | None = None) -> Path:
    """Merge the official-implementation LCB semstamp shard files (dedupes by
    task_id; sub-shard files like .shardXofN.subYofM.json are included).
    With impl set, uses the current naming
    semstamp_livecodebench_<impl>_<tag>.shardIofN.json; otherwise the
    historical pre-reorg semstamp_lcb_<tag>.shardIofN.json."""
    import dataloader

    ds = dataloader.get("livecodebench")
    stem = (f"semstamp_livecodebench_{impl}_{tag}" if impl
            else f"semstamp_lcb_{tag}")
    results, meta, seen = [], None, set()
    for i in range(num_shards):
        paths = sorted(out_dir("semstamp", "livecodebench").glob(f"{stem}.shard{i}of{num_shards}*.json"))
        for p in paths:
            with open(p) as f:
                d = json.load(f)
            meta = meta or d["meta"]
            for r in d["results"]:
                if r["task_id"] not in seen:
                    seen.add(r["task_id"])
                    results.append(r)
    order = {ds.task_id(t): i for i, t in enumerate(ds.load_tasks())}
    missing = set(order) - {r["task_id"] for r in results}
    if missing:
        print(f"WARNING: {len(missing)} tasks missing, e.g. {sorted(missing)[:5]}")
    results.sort(key=lambda r: order[r["task_id"]])
    meta = dict(meta)
    meta["shard"] = None
    meta["num_tasks"] = len(results)
    n_wm = sum(r["watermarked"]["detection"]["is_watermarked"] for r in results)
    meta["summary"] = {"detect_rate_watermarked": n_wm / len(results)}
    out = out_dir("semstamp", "livecodebench") / f"{stem}.json"
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"Merged {len(results)} tasks, TPR={n_wm}/{len(results)} -> {out}")
    return out


def auroc(pos_z: list, neg_z: list) -> float | None:
    """AUROC of z-score separation (watermarked = positive class), computed
    as the Mann-Whitney U statistic with average ranks for ties. None
    z-scores (detector had too few segments/tokens) are dropped."""
    import numpy as np

    pos = np.array([z for z in pos_z if z is not None], dtype=float)
    neg = np.array([z for z in neg_z if z is not None], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    both = np.concatenate([pos, neg])
    order = both.argsort()
    ranks = np.empty(len(both))
    ranks[order] = np.arange(1, len(both) + 1)
    # average ranks for ties
    for v in np.unique(both):
        m = both == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def summarize_all(tag: str = "qwen3.5-4b") -> None:
    """One line per inference JSON under out/: n, TPR, FP, z stats, AUROC,
    tokens, plus any pass@1 summaries sitting next to them. Detection
    verdicts go through detect_flag (is_watermarked with `prediction`
    fallback). AUROC pairs watermarked z against the same file's
    unwatermarked branch, falling back to the sibling *_control_detection
    file (SemStamp)."""
    import numpy as np

    def zs_of(dets):
        return [d.get("z_score") for d in dets]

    def zstats(dets):
        zs = [z for z in zs_of(dets) if z is not None]
        if not zs:
            return "z=n/a"
        return f"z mean={np.mean(zs):6.2f} med={np.median(zs):6.2f}"

    for path in sorted(OUT_DIR.glob("*/*/*.json")) + sorted(OUT_DIR.glob("*/*/markllm/*.json")):
        if path.name.startswith("pass1") or path.parent.name == "similarity":
            continue
        with open(path) as f:
            data = json.load(f)
        if "results" not in data or not data["results"]:
            continue
        rs = data["results"]
        rel = path.relative_to(OUT_DIR)
        if "control_detection" in path.name:
            dets = [r["detection"] for r in rs]
            n_fp = sum(detect_flag(d) for d in dets)
            print(f"{str(rel):68s} n={len(rs):3d} FP={n_fp/len(rs):6.1%} {zstats(dets)}")
            continue
        if "watermarked" not in rs[0]:
            continue
        wm_dets = [r["watermarked"]["detection"] for r in rs]
        n_wm = sum(detect_flag(d) for d in wm_dets)
        line = f"{str(rel):68s} n={len(rs):3d} TPR={n_wm/len(rs):6.1%} {zstats(wm_dets)}"
        base_key = ("unwatermarked" if "unwatermarked" in rs[0]
                    else "unwatermarked_control" if "unwatermarked_control" in rs[0] else None)
        base_dets = []
        if base_key:
            base_dets = [r[base_key]["detection"] for r in rs if base_key in r]
        if not base_dets:
            ctrl_path = path.with_name(path.stem.replace(f"_{tag}", "") + "_control_detection.json")
            if ctrl_path.exists():
                with open(ctrl_path) as f:
                    base_dets = [r["detection"] for r in json.load(f)["results"]]
                base_key = "control file"
        if base_dets:
            n_fp = sum(detect_flag(d) for d in base_dets)
            line += f" | base FP={n_fp/len(base_dets):6.1%} {zstats(base_dets)}"
            a = auroc(zs_of(wm_dets), zs_of(base_dets))
            if a is not None:
                line += f" | AUROC={a:.3f}"
        toks = [r["watermarked"].get("num_gen_tokens") for r in rs]
        toks = [t for t in toks if t is not None]
        if toks:
            line += f" | wm tokens mean={np.mean(toks):.0f}"
        print(line)

    for p in sorted(OUT_DIR.glob("*/*/pass1*summary*.json")):
        with open(p) as f:
            print(f"{str(p.relative_to(OUT_DIR)):68s} {json.dumps(json.load(f))}")


def summarize_lcb(tag: str = "qwen3.5-4b") -> None:
    """Print the summary table for the official-implementation LCB runs."""
    import numpy as np

    def show(name, zs, preds, extra=""):
        zs = [z for z in zs if z is not None]
        print(
            f"{name:34s} n={len(preds):3d} detect={sum(preds)/len(preds):6.1%} "
            f"mean z={np.mean(zs):6.2f} median z={np.median(zs):6.2f} {extra}"
        )

    kgw_p = out_dir("kgw", "livecodebench") / f"kgw_lcb_{tag}.json"
    if kgw_p.exists():
        d = json.load(open(kgw_p))
        rs = d["results"]
        for field in ("watermarked", "unwatermarked"):
            show(
                f"KGW {field}",
                [r[field]["detection"].get("z_score") for r in rs],
                [bool(r[field]["detection"].get("prediction")) for r in rs],
                extra=f"mean tokens={np.mean([r[field]['num_gen_tokens'] for r in rs]):.0f}",
            )

    ss_p = out_dir("semstamp", "livecodebench") / f"semstamp_lcb_{tag}.json"
    if ss_p.exists():
        d = json.load(open(ss_p))
        rs = d["results"]
        show(
            "SemStamp watermarked",
            [r["watermarked"]["detection"]["z_score"] for r in rs],
            [r["watermarked"]["detection"]["is_watermarked"] for r in rs],
            extra=(
                f"mean sents={np.mean([r['watermarked']['detection']['n_test_sents'] for r in rs]):.1f} "
                f"mean trials={np.mean([r['watermarked']['sampling_stats']['total_trials'] for r in rs]):.0f} "
                f"mean secs={np.mean([r['watermarked']['gen_seconds'] for r in rs]):.0f}"
            ),
        )

    ctrl_p = out_dir("semstamp", "livecodebench") / "semstamp_lcb_control_detection.json"
    if ctrl_p.exists():
        d = json.load(open(ctrl_p))
        rs = d["results"]
        show(
            "SemStamp control (KGW unwm)",
            [r["detection"]["z_score"] for r in rs],
            [r["detection"]["is_watermarked"] for r in rs],
        )

    for p in sorted(out_dir("kgw", "livecodebench").glob("pass1_*.json")) + sorted(
        out_dir("semstamp", "livecodebench").glob("pass1_*.json")
    ):
        d = json.load(open(p))
        if "summary" in d:
            print(f"{p.name}: {json.dumps(d['summary'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command",
        choices=["merge-semstamp", "analyze-semstamp", "merge-semstamp-lcb", "summarize-lcb",
                 "summarize"],
    )
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--model-tag", default="qwen3.5-4b")
    ap.add_argument("--impl", choices=["pilot", "official"], default=None,
                    help="merge-semstamp-lcb: current-naming shards for this impl "
                    "(omit for historical pre-reorg files)")
    args = ap.parse_args()
    if args.command == "merge-semstamp":
        merge_semstamp_shards(args.dataset, args.num_shards or 4, args.model_tag)
    elif args.command == "analyze-semstamp":
        analyze_semstamp(args.dataset, args.model_tag)
    elif args.command == "merge-semstamp-lcb":
        merge_semstamp_lcb_shards(args.num_shards or 6, args.model_tag, args.impl)
    elif args.command == "summarize":
        summarize_all(args.model_tag)
    else:
        summarize_lcb(args.model_tag)
