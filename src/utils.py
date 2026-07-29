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
import time
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


def task_chat_text(tokenizer, ds, task: dict, allow_comments: bool = False) -> str:
    """Chat-template text for one benchmark task (ds: a dataloader module)."""
    import dataloader  # lazy, as elsewhere in this module (MarkLLM `utils` clash)

    return chat_text(
        tokenizer,
        ds.build_prompt(task),
        dataloader.system_prompt(ds, allow_comments),
    )


def assert_resumable(prior_meta: dict, meta: dict, out_path) -> None:
    """Refuse to append to a run produced under a different configuration.

    Runners resume by task_id from whatever file sits at the output path, so a
    changed model / delta / gate / prompt policy / sampling setting would append
    incompatible completions to an existing file and relabel the whole thing
    with only the latest invocation's metadata. Compare the fields that define
    the experiment and abort loudly instead. Keys absent from the stored meta
    (files written before a field existed) are skipped rather than treated as
    mismatches, so historical outputs stay resumable.
    """
    if not prior_meta:
        return
    mismatched = []
    # detection_scope is compared even when ABSENT from the stored meta: a file
    # without it predates the extracted-code scoping and holds z-scores over the
    # raw completion, which cannot be pooled with new ones. Every other key
    # tolerates absence (see docstring) — this one must not.
    prior_scope = prior_meta.get("detection_scope", "raw_completion")
    if "detection_scope" in meta and prior_scope != meta["detection_scope"]:
        mismatched.append(("detection_scope", prior_scope, meta["detection_scope"]))
    for key in ("model", "dataset", "impl", "allow_comments", "system_prompt",
                "generation", "seed"):
        if key in prior_meta and prior_meta[key] != meta.get(key):
            mismatched.append((key, prior_meta[key], meta.get(key)))
    prior_wm, cur_wm = prior_meta.get("watermark", {}), meta.get("watermark", {})
    for key in ("delta", "gamma", "switch_threshold", "entropy_threshold",
                "context_width", "hash_key", "code_model", "variant"):
        if key in prior_wm and prior_wm[key] != cur_wm.get(key):
            mismatched.append((f"watermark.{key}", prior_wm[key], cur_wm.get(key)))
    if mismatched:
        detail = "\n".join(f"  {k}: existing={o!r} requested={n!r}" for k, o, n in mismatched)
        raise SystemExit(
            f"refusing to resume {out_path}\nit was produced under a different "
            f"configuration:\n{detail}\nuse a different --output, or delete the file."
        )


def extract_code(completion: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    # unclosed block (generation hit the token cap before the closing fence)
    m = re.search(r"```(?:python)?\n(.*)$", completion, re.DOTALL)
    return m.group(1).strip() if m else None


# ------------------------------------------------------------------ runners


def extract_program(completion: str | None) -> str | None:
    """The generated program to score, or None when the completion has none.

    Prefers the fenced block. Falls back to the whole completion ONLY when it
    parses as Python by itself: such a completion has no fence and no prose, so
    it IS the program and scoring it introduces none of the contamination that
    scoping to code exists to remove. Anything else is genuinely unscoreable and
    must not be quietly scored as if it were code — on MBPP this fallback is the
    difference between dropping 22% of samples and keeping them.
    """
    code = extract_code(completion)
    if code:
        return code
    if not completion or not completion.strip():
        return None
    import ast

    stripped = completion.strip()
    try:
        ast.parse(stripped)
    except SyntaxError:
        return None
    return stripped


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
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="models with custom hub code (e.g. OpenCoder)")


def load_hf_model(model_name: str, device: str, padding_side: str | None = None,
                  trust_remote_code: bool = False):
    """Standard HF tokenizer+model load (bf16). Returns (tokenizer, model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"padding_side": padding_side} if padding_side else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, **kw
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=trust_remote_code,
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
        # repro files store None for tasks whose detection was invalid
        return [d.get("z_score") for d in dets if d]

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
            n_fp = sum(detect_flag(d) for d in dets if d)
            print(f"{str(rel):68s} n={len(rs):3d} FP={n_fp/len(rs):6.1%} {zstats(dets)}")
            continue
        if "watermarked" not in rs[0]:
            continue
        wm_dets = [r["watermarked"]["detection"] for r in rs]
        n_wm = sum(detect_flag(d) for d in wm_dets if d)
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
            n_fp = sum(detect_flag(d) for d in base_dets if d)
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


def tpr_at_fpr(pos_z: list, neg_z: list, target_fpr: float) -> float:
    """Upstream analyze_results' ROC operating points (get_roc_auc/get_tpr):
    the best TPR achievable with FPR <= target, thresholding z-scores
    (positives = watermarked, negatives = human/base)."""
    import numpy as np

    pos = np.array([z for z in pos_z if z is not None], dtype=float)
    neg = np.array([z for z in neg_z if z is not None], dtype=float)
    best = 0.0
    for t in np.unique(np.concatenate([pos, neg])):
        fpr = (neg >= t).mean()
        if fpr <= target_fpr:
            best = max(best, (pos >= t).mean())
    return best


# ------------------------------------------------- unified evaluation protocol
#
# Every method's runner records and scores a run the same way, so numbers are
# comparable across methods and datasets. The three rules, all established on
# CodeWM (2026-07-24/25) and applied uniformly since:
#
#   1. DETECTION scores the extracted PROGRAM (`extract_program`), never the raw
#      completion — a chat model wraps code in a markdown fence and adds prose,
#      and scoring those non-code tokens inflates the green fraction on the
#      watermarked and unwatermarked classes alike, crushing their separation.
#      This rule governs the WATERMARK task only, which is ours to define.
#      **pass@1 is not ours to redefine** and keeps the benchmark's own
#      evaluation: `extracted_code` still holds `extract_code(completion)` and
#      `eval_pass1.py` still reads it, so a change on the watermark side can
#      never move a quality number. The two extractors agree whenever the model
#      emitted a fenced block, and differ only on the no-fence completions that
#      `extract_program` recovers by parsing — recovered for detection, still
#      counted the benchmark's way for pass@1;
#   2. the negative class is the PAIRED UNWATERMARKED completion — same model,
#      same prompt, watermark off. Not the benchmark's reference solution: a
#      human solution differs from model output in length, style and formatting,
#      so scoring against it conflates "is watermarked" with "was written by a
#      model" and reads systematically higher (measured on CodeWM/deepseek MBPP:
#      AUROC 0.798 vs human against 0.673 vs the paired baseline). The paired
#      control isolates the watermark and is the only class reported;
#   3. the headline metric is TPR at fixed FPR from the ROC, not the
#      z>threshold flip rate, which is near-0 by construction on short
#      completions regardless of how separable the classes are.
#
# `meta.detection_scope` records rule 1 per file; `assert_resumable` refuses to
# pool a file that predates it with one that follows it.
# value stamped into meta.detection_scope; deliberately NOT the string
# "extracted_code", which now names the pass@1 field
DETECTION_SCOPE = "extracted_program"


def branch_record(completion: str, num_gen_tokens: int | None, detect_program,
                  prompt: str | None = None) -> dict:
    """One generated branch (watermarked / unwatermarked) under the protocol.

    `detect_program(text, prompt)` is the method's scorer. The prompt is passed
    to every method but used only by SemStamp, whose chain z-test anchors its
    first segment against it; the token-level methods ignore it.
    """
    # `extracted_code` = the benchmark's own extraction, consumed by pass@1;
    # `program` = the detection input. Identical whenever a fence was emitted.
    code = extract_code(completion)
    program = extract_program(completion)
    rec = {
        "completion": completion,
        "extracted_code": code,
        # which text the z-test saw, so a reader never has to guess (the text
        # itself is recomputable from `completion` via extract_program)
        "detection_input": ("extracted_code" if code
                            else "parsed_completion" if program else None),
        "detection": detect_program(program, prompt),
    }
    if num_gen_tokens is not None:
        rec["num_gen_tokens"] = num_gen_tokens
    return rec


def rescore_results(results: list, detect_program, prompt_of=None) -> list:
    """Re-detect stored completions under the current protocol, in place.

    Generation is not repeated — only `detection` (and the `detection_input`
    label) is recomputed, from the `completion` text every runner already
    stores. `extracted_code` is rewritten with the benchmark's own extractor,
    which is what it always held, so pass@1 over a rescored file is unchanged. This
    brings a pre-protocol file onto the current rules without spending the
    generation budget again, and it is the ONLY honest way to compare an old run
    with a new one (see `assert_resumable`: the two scopes must never be pooled).

    Branches are discovered rather than hardcoded, so SemStamp's
    `unwatermarked_control` is picked up alongside the usual pair. Any `human`
    branch left over from the earlier protocol is dropped: the negative class is
    the paired unwatermarked completion (rule 2).
    """
    out = []
    for r in results:
        rec = {k: v for k, v in r.items() if k != "human"}
        for br in ("watermarked", "unwatermarked", "unwatermarked_control"):
            old = r.get(br)
            if not isinstance(old, dict) or "completion" not in old:
                continue
            prompt = prompt_of(r) if prompt_of else None
            new = branch_record(old["completion"], old.get("num_gen_tokens"),
                                detect_program, prompt)
            # keep whatever else the runner recorded (sampling_stats, timings)
            rec[br] = {**old, **new}
        out.append(rec)
    return out


def rescore_args_from_meta(args, meta: dict, keys) -> list:
    """Point `args` at the watermark config the FILE was produced with.

    Detection knobs (gate thresholds, gamma, checkpoint) must match the run being
    rescored, not today's CLI defaults — otherwise the rescore silently mixes a
    protocol change with a configuration change and the delta means nothing.
    Returns the list of overrides applied, for logging.
    """
    wm = meta.get("watermark", {})
    applied = []
    for key in keys:
        if wm.get(key) is None:
            continue
        attr = key.replace("-", "_")
        if getattr(args, attr, None) != wm[key]:
            applied.append(f"{attr}: {getattr(args, attr, None)} -> {wm[key]}")
            setattr(args, attr, wm[key])
    return applied


def rescore_meta(meta: dict, src, results: list) -> dict:
    """Metadata for a rescored copy: the original's, restamped."""
    out = dict(meta)
    out["detection_scope"] = DETECTION_SCOPE
    out["rescored_from"] = str(src)
    out["rescored_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["summary"] = detection_summary(results)
    return out


def rescore_path(src) -> Path:
    """Where a rescored copy goes. Never overwrites the original — result JSONs
    are append-only evidence (CLAUDE.md)."""
    src = Path(src)
    return src.with_name(src.stem + ".rescored.json")


def z_scores(dets) -> list:
    """z-scores from an iterable of detection dicts, skipping missing/invalid
    ones (a detection is None when there was no code to score)."""
    return [
        d["z_score"] for d in dets
        if d and not d.get("invalid") and d.get("z_score") is not None
    ]


def _roc_block(pos_z: list, neg_z: list) -> dict:
    """AUROC + TPR at fixed operating FPRs, against the paired unwatermarked
    class. `None` throughout when either class is empty, rather than a
    fabricated 0."""
    if not pos_z or not neg_z:
        return {"auroc": None, "tpr@0pct": None, "tpr@1pct": None, "tpr@5pct": None,
                "n_pos": len(pos_z), "n_neg": len(neg_z)}
    return {
        "auroc": auroc(pos_z, neg_z),
        "tpr@0pct": tpr_at_fpr(pos_z, neg_z, 0.0),
        "tpr@1pct": tpr_at_fpr(pos_z, neg_z, 0.01),
        "tpr@5pct": tpr_at_fpr(pos_z, neg_z, 0.05),
        "n_pos": len(pos_z),
        "n_neg": len(neg_z),
    }


def detection_summary(results: list) -> dict:
    """Summary block for any watermark run (see "unified evaluation protocol").

    The headline number is TPR@FPR from the ROC — NOT the z>threshold flip rate.
    Those are wildly different statistics on code: the z-test needs many scored
    tokens to clear a fixed threshold, so a short completion can carry a
    perfectly detectable green-fraction bias and still never reach z>4. The flip
    rate is kept as `detect_rate_watermarked` for continuity with older files.
    The negative class is the paired unwatermarked completion (rule 2).
    """
    # SemStamp names its non-watermarked class `unwatermarked_control` (it
    # borrows completions from a KGW run rather than generating its own)
    def _base(r):
        b = r.get("unwatermarked") or r.get("unwatermarked_control") or {}
        return b.get("detection")

    wm_z = z_scores(r["watermarked"]["detection"] for r in results)
    base_z = z_scores(_base(r) for r in results)
    # samples whose completion had no fenced code block are unscoreable by
    # construction; report the rate so it is never mistaken for weak detection
    n_unscoreable = sum(r["watermarked"]["detection"] is None for r in results)
    # detection is None when the branch produced nothing scoreable
    flag = lambda d: bool(d) and detect_flag(d)
    n_wm = sum(flag(r["watermarked"]["detection"]) for r in results)
    n_fp = sum(flag(_base(r)) for r in results)
    return {
        "n": len(results),
        "n_unscoreable_watermarked": n_unscoreable,
        "detect_rate_watermarked": n_wm / max(len(results), 1),
        "false_positive_rate_unwatermarked": n_fp / max(len(results), 1),
        "mean_z_watermarked": sum(wm_z) / len(wm_z) if wm_z else None,
        "mean_z_unwatermarked": sum(base_z) / len(base_z) if base_z else None,
        **_roc_block(wm_z, base_z),
    }


def fmt_pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def summarize_method(method: str = "codewm", impl: str | None = None) -> None:
    """Markdown table for every run of one method, on the unified metric.

    The headline is TPR@FPR from the ROC (what the paper reports), not the
    z>threshold flip rate; both are shown because the flip rate is what older
    files recorded and it is near-0 by construction on short completions.
    `variant` is the gate operating point: -10 = the released run.sh/main.py
    default (gate always open, every token embedded+scored); 0.7 = the README's
    WLLM example setting, which gates ~90% of tokens away.

    Metrics are recomputed from the per-task detections rather than read from
    `meta.summary`, so files written before the summary block existed are
    included on equal terms.
    """
    rows = []
    for path in sorted(OUT_DIR.glob(f"{method}/*/{method}_*.json")):
        if ".shard" in path.name or path.name.startswith("pass1_"):
            continue
        file_impl = next((k for k in ("repro", "official", "pilot", "port")
                          if f"_{k}_" in path.name or path.name.endswith(f"_{k}.json")),
                         "markllm" if "markllm" in str(path) else "port")
        if impl and file_impl != impl:
            continue
        with open(path) as f:
            data = json.load(f)
        results = data.get("results") or []
        # repro marks tasks whose watermarked/human detection was unscoreable
        rs = [r for r in results if r.get("valid", True)]
        if not rs or "watermarked" not in rs[0]:
            continue
        s = detection_summary(rs)
        if s["auroc"] is None:
            continue
        wm_cfg = data["meta"].get("watermark", {})
        # a compact label from whatever knobs this method actually has
        bits = []
        if wm_cfg.get("variant"):
            bits.append(str(wm_cfg["variant"]))
        for key, short in (("switch_threshold", "sw"), ("entropy_threshold", "H"),
                           ("gamma", "g")):
            if wm_cfg.get(key) is not None:
                bits.append(f"{short}={wm_cfg[key]:g}")
        variant = " ".join(bits) or wm_cfg.get("algorithm", "?")
        # repro always scored the completion ids sliced past the prompt; port
        # files without the key predate extracted-code scoping (raw completion,
        # fence + prose included) and are NOT comparable to the ones with it
        scope = data["meta"].get(
            "detection_scope", "completion_ids" if file_impl == "repro" else "raw"
        )
        rows.append({
            "cell": path.stem.replace(f"{method}_", ""),
            "model": data["meta"]["model"].split("/")[-1],
            "dataset": data["meta"].get("dataset", "?"),
            "impl": file_impl,
            "delta": wm_cfg.get("delta"),
            "variant": variant,
            "scope": scope,
            "summary": s,
        })
    rows.sort(key=lambda r: (r["impl"], r["variant"], r["model"].lower(),
                             r["dataset"], r["cell"]))

    print("| cell | impl | variant | δ | scored over | n | z>thr rate | AUROC | "
          "TPR@1% | TPR@5% | mean z wm | mean z base |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    f2 = lambda v: "n/a" if v is None else f"{v:+.2f}"
    for r in rows:
        s = r["summary"]
        print(f"| {r['cell']} | {r['impl']} | {r['variant']} | {r['delta']} | "
              f"{r['scope']} | {s['n']} | {s['detect_rate_watermarked']:.1%} | "
              f"{f3(s['auroc'])} | {fmt_pct(s['tpr@1pct'])} | {fmt_pct(s['tpr@5pct'])} | "
              f"{f2(s['mean_z_watermarked'])} | {f2(s['mean_z_unwatermarked'])} |")
    print()
    if any(r["scope"] == "raw" for r in rows):
        print("NOTE: rows marked scope=raw predate the program-scoped detection fix "
              "— their z-scores were computed over the whole completion (markdown "
              "fence and prose included) and are NOT comparable with scope="
              "extracted_program rows. Rescore or regenerate to compare.")
        print()
    print("TPR@FPR 0% / 1% / 5%, positives vs the paired unwatermarked completions:")
    for r in rows:
        s = r["summary"]
        print(f"  {r['cell']:62s} {fmt_pct(s['tpr@0pct']):>6s} /"
              f"{fmt_pct(s['tpr@1pct']):>6s} /{fmt_pct(s['tpr@5pct']):>6s}"
              f"   (n_pos={s['n_pos']}, n_neg={s['n_neg']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command",
        choices=["merge-semstamp", "analyze-semstamp", "merge-semstamp-lcb", "summarize-lcb",
                 "summarize", "summarize-methods"],
    )
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--model-tag", default="qwen3.5-4b")
    ap.add_argument("--method", default="all",
                    choices=["all", "kgw", "semstamp", "sweet", "codewm"],
                    help="summarize-methods: which method's runs to table")
    ap.add_argument("--impl", choices=["pilot", "official", "port", "repro"], default=None,
                    help="merge-semstamp-lcb: current-naming shards for this impl "
                    "(omit for historical pre-reorg files). summarize-codewm: "
                    "restrict to port or repro runs")
    args = ap.parse_args()
    if args.command == "merge-semstamp":
        merge_semstamp_shards(args.dataset, args.num_shards or 4, args.model_tag)
    elif args.command == "analyze-semstamp":
        analyze_semstamp(args.dataset, args.model_tag)
    elif args.command == "merge-semstamp-lcb":
        merge_semstamp_lcb_shards(args.num_shards or 6, args.model_tag, args.impl)
    elif args.command == "summarize":
        summarize_all(args.model_tag)
    elif args.command == "summarize-methods":
        for m in (["kgw", "semstamp", "sweet", "codewm"] if args.method == "all"
                  else [args.method]):
            print(f"\n########## {m}")
            summarize_method(m, args.impl if args.impl in ("port", "repro") else None)
    else:
        summarize_lcb(args.model_tag)
