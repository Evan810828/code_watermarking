"""Aggregate the MarkLLM+vLLM reproduction outputs into a comparison table.

Merges semstamp shard files (in task order), computes detection/z stats per
run, and prints a pilot-vs-repro summary. Run after all repro jobs finish.
Writes out/repro_summary.json (spans both methods, so it sits at out/ root).
"""

import glob
import json

import numpy as np

from adapter import load_pilot_utils

pilot_utils = load_pilot_utils()


def merge_shards(ss_dir, pattern: str, merged_path):
    paths = sorted(glob.glob(str(ss_dir / pattern)))
    if not paths:
        return None
    results, meta = [], None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        meta = meta or d["meta"]
        results.extend(d["results"])
    results.sort(key=lambda r: str(r["task_id"]))
    n_wm = sum(r["watermarked"]["detection"]["is_watermarked"] for r in results)
    ctrl = [r for r in results if "unwatermarked_control" in r]
    n_fp = sum(r["unwatermarked_control"]["detection"]["is_watermarked"] for r in ctrl)
    meta = dict(meta)
    meta.pop("shard", None)
    meta["num_tasks"] = len(results)
    meta["summary"] = {
        "detect_rate_watermarked": n_wm / len(results),
        "false_positive_rate_unwatermarked": n_fp / len(ctrl) if ctrl else None,
    }
    with open(merged_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    return merged_path


def stats(path, ctrl_branch="unwatermarked_control"):
    with open(path) as f:
        d = json.load(f)
    rs = d["results"]
    wm_z = [r["watermarked"]["detection"].get("z_score") for r in rs]
    wm_z = [z for z in wm_z if z is not None]
    wm_det = np.mean([r["watermarked"]["detection"]["is_watermarked"] for r in rs])
    ctrl = [r[ctrl_branch]["detection"] for r in rs if ctrl_branch in r]
    fp = np.mean([c["is_watermarked"] for c in ctrl]) if ctrl else None
    out = {
        "n": len(rs),
        "TPR": float(wm_det),
        "FP": None if fp is None else float(fp),
        "mean_z_wm": float(np.mean(wm_z)) if wm_z else None,
    }
    if "sampling_stats" in rs[0]["watermarked"]:
        segs = [s for r in rs for s in r["watermarked"].get("segment_log", [])]
        if segs:
            one_sig = np.mean([s["distinct_candidate_sigs"] == 1 for s in segs])
            maxed = np.mean([s["status"] == "maxedout" for s in segs])
            out["frac_segments_1sig"] = float(one_sig)
            out["frac_segments_maxedout"] = float(maxed)
    return out


def main():
    for ds in ["humaneval", "mbpp"]:
        ss_dir = pilot_utils.out_dir("semstamp", ds, "markllm")
        for seg in ["line", "sent"]:
            merged = ss_dir / f"semstamp_vllm_{seg}_{ds}_qwen3.5-4b.json"
            if not merged.exists() or glob.glob(str(ss_dir / f"semstamp_vllm_{seg}_{ds}_qwen3.5-4b.shard*.json")):
                merge_shards(ss_dir, f"semstamp_vllm_{seg}_{ds}_qwen3.5-4b.shard*.json", merged) or (
                    merged if merged.exists() else None
                )

    report = {}
    for ds in ["humaneval", "mbpp"]:
        report[ds] = {}
        kgw_dir = pilot_utils.out_dir("kgw", ds, "markllm")
        ss_dir = pilot_utils.out_dir("semstamp", ds, "markllm")
        p = kgw_dir / f"kgw_vllm_{ds}_qwen3.5-4b.json"
        if p.exists():
            report[ds]["kgw_vllm"] = stats(p, ctrl_branch="unwatermarked")
        for seg in ["line", "sent"]:
            p = ss_dir / f"semstamp_vllm_{seg}_{ds}_qwen3.5-4b.json"
            if p.exists():
                report[ds][f"semstamp_vllm_{seg}"] = stats(p)
        p = ss_dir / f"crosscheck_semstamp_{ds}.json"
        if p.exists():
            with open(p) as f:
                report[ds]["detection_crosscheck"] = json.load(f)["summary"]

    out_path = pilot_utils.OUT_DIR / "repro_summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
