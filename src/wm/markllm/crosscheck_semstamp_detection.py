"""Cross-validate the pilot's SemStamp detector with the MarkLLM package.

Runs MarkLLM's SemStamp detection (cached-embedder version of
SemStamp.detect_watermark, see adapter.py) over the pilot's existing
semstamp outputs — watermarked completions and unwatermarked controls — with
both segmenters:

- line: same segmentation the pilot used -> z should match the pilot's stored
  z-scores (the LSH normals, signature and mask derivations were verified
  bit-identical between the pilot code and MarkLLM/nearpy).
- sent: MarkLLM's original sent_tokenize -> shows what the unmodified package
  reports on code.

Usage (from this directory):
    python crosscheck_semstamp_detection.py --dataset humaneval
"""

import argparse
import json

import numpy as np

from adapter import MarkLLMSemStamp, load_pilot_utils

pilot_utils = load_pilot_utils()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--model-tag", default="qwen3.5-4b")
    args = ap.parse_args()

    with open(
        pilot_utils.out_dir("semstamp", args.dataset)
        / f"semstamp_{args.dataset}_{args.model_tag}.json"
    ) as f:
        results = json.load(f)["results"]

    ss = MarkLLMSemStamp(device="cpu")  # pilot used a CPU generator for masks

    rows = []
    for r in results:
        row = {"task_id": r["task_id"], "pilot_z": r["watermarked"]["detection"]["z_score"]}
        row["markllm_line"] = ss.detect(r["watermarked"]["completion"], "line")
        row["markllm_sent"] = ss.detect(r["watermarked"]["completion"], "sent")
        if "unwatermarked_control" in r:
            ctrl = r["unwatermarked_control"]["completion"]
            row["pilot_ctrl_z"] = r["unwatermarked_control"]["detection"]["z_score"]
            row["ctrl_markllm_line"] = ss.detect(ctrl, "line")
            row["ctrl_markllm_sent"] = ss.detect(ctrl, "sent")
        rows.append(row)

    out_path = (
        pilot_utils.out_dir("semstamp", args.dataset, "markllm")
        / f"crosscheck_semstamp_{args.dataset}.json"
    )

    # agreement between pilot z and MarkLLM line-mode z
    both = [
        (row["pilot_z"], row["markllm_line"]["z_score"])
        for row in rows
        if row["pilot_z"] is not None and row["markllm_line"]["z_score"] is not None
    ]
    diffs = [abs(a - b) for a, b in both]
    line_tpr = np.mean([row["markllm_line"]["is_watermarked"] for row in rows])
    sent_tpr = np.mean([row["markllm_sent"]["is_watermarked"] for row in rows])
    sent_testable = np.mean([row["markllm_sent"]["n_tested"] > 0 for row in rows])
    sent_ntest = np.mean([row["markllm_sent"]["n_tested"] for row in rows])
    ctrl_line_fp = np.mean(
        [row["ctrl_markllm_line"]["is_watermarked"] for row in rows if "ctrl_markllm_line" in row]
    )
    ctrl_sent_fp = np.mean(
        [row["ctrl_markllm_sent"]["is_watermarked"] for row in rows if "ctrl_markllm_sent" in row]
    )
    summary = {
        "n": len(rows),
        "line_vs_pilot": {
            "n_compared": len(both),
            "mean_abs_z_diff": float(np.mean(diffs)),
            "max_abs_z_diff": float(np.max(diffs)),
            "exact_match_rate(<1e-9)": float(np.mean([d < 1e-9 for d in diffs])),
        },
        "markllm_line": {"tpr": float(line_tpr), "fp": float(ctrl_line_fp)},
        "markllm_sent": {
            "tpr": float(sent_tpr),
            "fp": float(ctrl_sent_fp),
            "mean_n_tested": float(sent_ntest),
            "frac_with_any_testable_segment": float(sent_testable),
        },
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
