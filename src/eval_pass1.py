"""Execution-based pass@1 evaluation for inference output JSONs, any dataset.

The dataset is read from the input file's meta; program construction and
execution live in the dataloader modules (dataloader.<ds>.evaluate), so this
script never needs dataset-specific code.

Output schema (unchanged from the original pilot):
    per-task: {"task_id", "<branch>": {"passed": bool, "reason": "pass|fail|timeout|no_code|..."}}
    summary : {"<dataset>": {"n", "pass@1_<branch>", "detail": {...}}}

Each program runs in a fresh python subprocess with a timeout. Executes
model-generated code — run only on trusted infrastructure.

Usage:
    python eval_pass1.py --input ../out/semstamp/humaneval/semstamp_humaneval_qwen3.5-4b.json --tag semstamp
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dataloader

BRANCHES = ["watermarked", "unwatermarked"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag", default=None, help="output name tag, e.g. 'semstamp'")
    ap.add_argument("--fields", nargs="+", default=BRANCHES)
    ap.add_argument("--timeout", type=float, default=None,
                    help="default: the dataset's EVAL_TIMEOUT")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default=None,
                    help="default: next to the inference JSON they evaluate")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    dataset = dataloader.normalize_name(data["meta"]["dataset"])
    ds = dataloader.get(dataset)
    timeout = args.timeout or ds.EVAL_TIMEOUT
    results = data["results"]
    branches = [b for b in args.fields if b in results[0]]
    print(f"{args.input}: {len(results)} {dataset} tasks, branches: {branches}")

    def eval_entry(r):
        # runners for slim datasets (LCB) store only task_id, which is all
        # their evaluate() needs
        task = r.get("task") or {"task_id": r["task_id"]}
        entry = {"task_id": r["task_id"]}
        for b in branches:
            code = r[b].get("extracted_code")
            if not code:
                entry[b] = {"passed": False, "reason": "no_code"}
                continue
            entry[b] = ds.evaluate(task, code, timeout)
        return entry

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        per_task = list(ex.map(eval_entry, results))

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.input).resolve().parent
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"pass1{tag}_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(per_task, f, indent=1)

    summary = {"n": len(per_task)}
    detail = {}
    for b in branches:
        # bucket reasons like "error@test3" under "error"
        reasons = [e[b]["reason"].split("@")[0] for e in per_task]
        detail[b] = {k: reasons.count(k) for k in sorted(set(reasons))}
        summary[f"pass@1_{b}"] = sum(e[b]["passed"] for e in per_task) / len(per_task)
    summary["detail"] = detail

    sum_path = out_dir / f"pass1{tag}_summary.json"
    existing = json.load(open(sum_path)) if sum_path.exists() else {}
    existing[dataset] = summary
    with open(sum_path, "w") as f:
        json.dump(existing, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> {out_path}\n-> {sum_path}")


if __name__ == "__main__":
    main()
