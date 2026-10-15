"""Render the verified full-split SeqMark ROC and exact FPR-budget functions."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import platform
import shutil
import socket
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[4]
LEDGER = Path(__file__).resolve().parent
PUBLISH = ROOT / "out/seqmark/figures"
VERIFICATION = ROOT / "out/seqmark/runs/20260914T225431Z_final_summary/manifest.json"
sys.path.insert(0, str(ROOT / "src"))
import utils

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

manifest_path = LEDGER / "manifest.json"
if manifest_path.exists():
    raise RuntimeError("A figure run record already exists; use a new run directory.")
reference = json.loads(VERIFICATION.read_text())
assert reference["status"] == "completed"
manifest = {
    "run_id": LEDGER.name, "study": "baseline_watermarking_seqmark",
    "stage": "roc_figure", "status": "running", "started_at": now(), "ended_at": None,
    "command": sys.argv,
    "code": {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "plot_source_sha256": sha(__file__), "metrics_source_sha256": sha(ROOT / "src/utils.py"),
        "generation_source_fingerprint": "952a90f4b7f79f76f05f07fb62aeea349b2008ba724fec0e312c683aacb3362d",
    },
    "resolved_configuration": {
        "datasets": ["humaneval", "mbpp"], "lsh_dim": 5, "gamma": 0.1,
        "center_samples": 50, "seed": 42, "score_scope": "extracted_program",
        "negative_class": "paired_unwatermarked", "exclude_unscoreable_per_class": True,
        "decision_rule": "score >= threshold",
        "top_panels": "empirical ROC vertices with conventional straight dashed connectors",
        "bottom_panels": "max TPR among vertices with FPR <= budget; no interpolation",
        "fpr_budgets": [0.01, 0.05], "full_axis_range": [0, 1], "budget_axis_range": [0, 0.12],
        "figure_inches": [7.2, 6.8], "raster_dpi": 300, "uncertainty": "one seed; no error bars",
    },
    "source_verification": str(VERIFICATION), "source_verification_sha256": sha(VERIFICATION),
    "source_results": reference["source_results"],
    "environment": {"host": socket.gethostname(), "python": platform.python_version(),
                    "matplotlib": matplotlib.__version__, "numpy": np.__version__},
}
write_json(manifest_path, manifest)

try:
    curves = {}
    metrics = {}
    point_rows = []
    readme = (ROOT / "out/README.md").read_text()
    first = readme.index("| method | impl | dataset | n | scored |", readme.index("## Results"))
    table = readme[first:readme.index("\n\n", first)].splitlines()[2:]
    for source in reference["source_results"]:
        path = Path(source["input"])
        assert sha(path) == source["input_sha256"], path
        data = json.loads(path.read_text())
        meta, results = data["meta"], data["results"]
        name, encoder = source["dataset"], source["encoder"]
        assert meta["watermark"]["lsh_dim"] == 5 and meta["watermark"]["gamma"] == 0.1
        assert meta["watermark"]["center_samples"] == 50
        assert meta["detection_scope"] == "extracted_program"
        assert len(results) == (164 if name == "humaneval" else 500)
        pos = utils.z_scores(r["watermarked"]["detection"] for r in results)
        neg = utils.z_scores(r["unwatermarked"]["detection"] for r in results)
        assert np.isfinite(pos + neg).all()
        scores = np.array(pos + neg)
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
        area = float(roc_auc_score(labels, scores))
        summary = utils.detection_summary(results)
        assert len(np.unique(scores)) == 2 and len(fpr) == 3
        assert math.isclose(area, summary["auroc"], rel_tol=0, abs_tol=1e-12)
        budget_metrics = {}
        for budget, key in ((0.01, "tpr@1pct"), (0.05, "tpr@5pct")):
            value = float(tpr[fpr <= budget].max())
            assert value == summary[key] == utils.tpr_at_fpr(pos, neg, budget)
            budget_metrics[key] = value
        label = "Gemma" if encoder.startswith("google/") else "SBERT"
        impl = "port (embeddinggemma-300m)" if label == "Gemma" else "port (SBERT)"
        matching = [r for r in table if [c.strip() for c in r.split("|")[1:4]] == ["SeqMark", impl, name]]
        assert len(matching) == 1
        cells = [c.strip() for c in matching[0].split("|")[1:-1]]
        assert cells[6:9] == [f"{area:.3f}", f"{budget_metrics['tpr@1pct']:.1%}", f"{budget_metrics['tpr@5pct']:.1%}"]
        curves[name, label] = {"fpr": fpr, "tpr": tpr, "auroc": area}
        metrics[encoder + "/" + name] = {"auroc": area, **budget_metrics, "n_pos": len(pos), "n_neg": len(neg)}
        for x, y, threshold in zip(fpr, tpr, thresholds):
            point_rows.append({
                "dataset": name, "encoder": encoder, "threshold": float(threshold),
                "false_positive_rate": float(x), "true_positive_rate": float(y),
                "false_positives": int(round(x * len(neg))), "true_positives": int(round(y * len(pos))),
                "scoreable_watermarked": len(pos), "scoreable_unwatermarked": len(neg),
            })

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "axes.edgecolor": "#555555",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.8))
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.165, top=0.855, wspace=0.29, hspace=0.48)
    fig.suptitle("SeqMark ROC and detection under low false-positive rates", fontsize=11.3, y=0.982)
    fig.text(0.5, 0.943, "Qwen3.5-4B  |  LSH dimensions = 5, gamma = 0.1  |  50 centre samples",
             ha="center", va="center", fontsize=8.4, color="#444444")
    styles = {
        "Gemma": {"color": "#0072B2", "marker": "o", "ms": 6.5},
        "SBERT": {"color": "#D55E00", "marker": "s", "ms": 4.8},
    }
    titles = [("humaneval", "HumanEval"), ("mbpp", "MBPP")]
    for col, (name, title) in enumerate(titles):
        top, low = axs[0, col], axs[1, col]
        top.set_title(f"({chr(97 + col)}) {title}: full ROC", loc="left", pad=8)
        top.plot([0, 100], [0, 100], color="#bbbbbb", lw=0.9, ls=":", zorder=1)
        for label, st in styles.items():
            c = curves[name, label]
            top.plot(c["fpr"] * 100, c["tpr"] * 100, linestyle="--", lw=1.25,
                     color=st["color"], marker=st["marker"], markersize=st["ms"],
                     markerfacecolor="white", markeredgewidth=1.4, clip_on=False,
                     label=f"{label} (AUROC {c['auroc']:.3f})", zorder=3)
        top.set(xlim=(0, 100), ylim=(0, 100), xlabel="False positive rate (%)")
        top.set_xticks([0, 20, 40, 60, 80, 100])
        top.set_yticks([0, 20, 40, 60, 80, 100])
        top.legend(loc="lower right", frameon=False, handlelength=2.2)

        low.set_title(f"({chr(99 + col)}) {title}: FPR budgets up to 12%", loc="left", pad=10)
        for budget in (1, 5):
            low.axvline(budget, color="#999999", lw=0.9, ls=":", zorder=1)
            low.text(budget, 8, f"{budget}%", ha="center", va="bottom", fontsize=8, color="#555555")
        for label, st in styles.items():
            c = curves[name, label]
            budgets = np.unique(np.r_[0.0, 0.01, 0.05, c["fpr"][c["fpr"] <= 0.12], 0.12])
            best = np.array([c["tpr"][c["fpr"] <= q].max() for q in budgets])
            low.step(budgets * 100, best * 100, where="post", lw=1.6, color=st["color"], zorder=3)
            actual = c["fpr"] <= 0.12
            low.plot(c["fpr"][actual] * 100, c["tpr"][actual] * 100,
                     linestyle="None", color=st["color"], marker=st["marker"], markersize=st["ms"],
                     markerfacecolor="white", markeredgewidth=1.4, clip_on=False, zorder=5)
            op_x, op_y = c["fpr"][1] * 100, c["tpr"][1] * 100
            if name == "humaneval":
                text_xy = (1.6, 70) if label == "Gemma" else (11.5, 49)
            else:
                text_xy = (3.0, 76) if label == "Gemma" else (11.5, 49)
            low.annotate(f"({op_x:.2f}%, {op_y:.2f}%)",
                         xy=(op_x, op_y), xytext=text_xy,
                         fontsize=7.8, color=st["color"], ha="left" if label == "Gemma" else "right",
                         arrowprops={"arrowstyle": "-", "color": st["color"], "lw": 0.7},
                         bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
        low.set(xlim=(0, 12), ylim=(0, 100), xlabel="Allowed false positive rate (%)")
        low.set_xticks([0, 1, 5, 10, 12])
        low.set_yticks([0, 20, 40, 60, 80, 100])
        for ax in (top, low):
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=3)
    axs[0, 0].set_ylabel("True positive rate (%)")
    axs[1, 0].set_ylabel("Best attainable true positive rate (%)")
    fig.text(0.105, 0.093,
             "Top: markers are attainable thresholds; dashed segments connect ROC vertices.\n"
             "Bottom: best TPR satisfying each FPR budget, with no interpolation or tie splitting.\n"
             "Scored watermarked / control: HumanEval 164 / 164; MBPP 500 / 493. Seed 42.",
             fontsize=7.8, va="top", linespacing=1.5, color="#444444")

    artifacts = []
    for suffix in ("pdf", "png"):
        output = LEDGER / f"seqmark_roc.{suffix}"
        fig.savefig(output, dpi=300)
        artifacts.append(output)
    plt.close(fig)
    point_path = LEDGER / "seqmark_roc_points.csv"
    with point_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    artifacts.append(point_path)
    PUBLISH.mkdir(parents=True, exist_ok=True)
    for output in artifacts:
        shutil.copy2(output, PUBLISH / output.name)
    write_json(LEDGER / "metrics.json", metrics)
    write_json(LEDGER / "diagnostics.json", {
        "input_hashes_match_verified_results": True, "metrics_match_primary_table": True,
        "shared_and_independent_tpr_match": True, "threshold_points_per_curve": 3,
        "score_values_per_curve": 2, "curve_count": len(curves),
    })
    manifest.update(status="completed", artifacts=[
        {"path": str(p), "sha256": sha(p), "published_path": str(PUBLISH / p.name)}
        for p in artifacts
    ])
    print(json.dumps({"artifacts": manifest["artifacts"], "metrics": metrics}, indent=2))
except BaseException as exc:
    manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    manifest["ended_at"] = now()
    write_json(manifest_path, manifest)
