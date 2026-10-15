"""Plot empirical z-score masses for the four verified full SeqMark runs."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import platform
import shutil
import socket
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LEDGER = Path(__file__).resolve().parent
PUBLISH = ROOT / "out/seqmark/figures"
VERIFICATION = ROOT / "out/seqmark/runs/20260914T225431Z_final_summary/manifest.json"
sys.path.insert(0, str(ROOT / "src"))
import utils

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

manifest_path = LEDGER / "manifest.json"
if manifest_path.exists():
    raise RuntimeError("Use a new directory for each figure execution.")
verification = json.loads(VERIFICATION.read_text())
assert verification["status"] == "completed"
manifest = {
    "run_id": LEDGER.name, "study": "baseline_watermarking_seqmark", "stage": "z_score_distribution",
    "status": "running", "started_at": now(), "ended_at": None, "command": sys.argv,
    "code": {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "plot_source_sha256": sha(__file__), "metrics_source_sha256": sha(ROOT / "src/utils.py")},
    "resolved_configuration": {
        "model": "Qwen/Qwen3.5-4B", "lsh_dim": 5, "gamma": 0.1, "center_samples": 50, "seed": 42,
        "datasets": ["humaneval", "mbpp"], "negative_class": "paired_unwatermarked",
        "scope": "extracted_program", "exclude_unscoreable_per_class": True,
        "distribution": "exact probability mass at each observed score; within-class normalization",
        "plot": "two score categories with side-by-side watermarked and control bars",
        "bar_labels": "percentage and count / scoreable class size",
        "smoothing": "none", "std_ddof": 0, "mean_std": "across task scores, not across seeds",
        "figure_inches": [7.2, 6.6], "dpi": 300, "y_axis_percent_limits": [0, 110],
    },
    "source_verification": str(VERIFICATION), "source_verification_sha256": sha(VERIFICATION),
    "source_results": verification["source_results"],
    "environment": {"host": socket.gethostname(), "python": platform.python_version(),
                    "matplotlib": matplotlib.__version__, "numpy": np.__version__},
}
save(manifest_path, manifest)
try:
    panels, exported, metrics = {}, [], {}
    for item in verification["source_results"]:
        path = Path(item["input"])
        assert sha(path) == item["input_sha256"], path
        data = json.loads(path.read_text())
        assert data["meta"]["watermark"]["lsh_dim"] == 5
        assert data["meta"]["watermark"]["gamma"] == 0.1
        assert data["meta"]["watermark"]["center_samples"] == 50
        dataset = item["dataset"]
        encoder = "Gemma" if item["encoder"].startswith("google/") else "SBERT"
        branches = {}
        for branch in ("watermarked", "unwatermarked"):
            scores = utils.z_scores(r[branch]["detection"] for r in data["results"])
            assert np.isfinite(scores).all()
            values, counts = np.unique(scores, return_counts=True)
            assert len(values) == 2 and np.allclose(values, [-1 / 3, 3.0], rtol=0, atol=1e-14)
            n = len(scores)
            masses = counts / n
            mean, std = statistics.mean(scores), statistics.pstdev(scores)
            reconstructed_mean = float(np.sum(values * masses))
            reconstructed_std = float(np.sqrt(np.sum(masses * (values - reconstructed_mean) ** 2)))
            assert math.isclose(mean, reconstructed_mean, rel_tol=0, abs_tol=1e-12)
            assert math.isclose(std, reconstructed_std, rel_tol=0, abs_tol=1e-12)
            assert counts.sum() == n and math.isclose(float(masses.sum()), 1.0)
            branches[branch] = {"counts": counts, "percent": masses * 100, "n": n, "mean": mean, "std": std}
            key = item["encoder"] + "/" + dataset + "/" + branch
            metrics[key] = {"n_scoreable": n, "mean_z": mean, "std_z_ddof0": std,
                            "z_values": values.tolist(), "counts": counts.tolist(),
                            "probability_mass": masses.tolist()}
            for value, count, mass in zip(values, counts, masses):
                exported.append({"dataset": dataset, "encoder": item["encoder"], "branch": branch,
                                 "z_score": float(value), "count": int(count), "n_scoreable": n,
                                 "probability_mass": float(mass), "mean_z": mean, "std_z_ddof0": std})
        assert branches["watermarked"]["n"] == (164 if dataset == "humaneval" else 500)
        assert branches["unwatermarked"]["n"] == (164 if dataset == "humaneval" else 493)
        panels[encoder, dataset] = branches

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555555", "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "hatch.linewidth": 0.55,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.15, top=0.785, hspace=0.60, wspace=0.28)
    fig.suptitle("SeqMark z-score distributions", fontsize=12.5, y=0.987)
    fig.text(0.5, 0.948, "Qwen3.5-4B  |  LSH dimensions = 5, gamma = 0.1  |  50 centre samples",
             ha="center", fontsize=8.5, color="#444444")
    colors = {"watermarked": "#0072B2", "unwatermarked": "#D55E00"}
    legend_handles = []
    for row, encoder in enumerate(("Gemma", "SBERT")):
        for col, (dataset, title) in enumerate((("humaneval", "HumanEval"), ("mbpp", "MBPP"))):
            ax = axes[row, col]
            branch_data = panels[encoder, dataset]
            ax.set_title(f"({chr(97 + 2 * row + col)}) {title} / {encoder}", loc="left", pad=29)
            for branch, shift, label, hatch in (
                ("watermarked", -0.18, "Watermarked (WM)", ""),
                ("unwatermarked", 0.18, "Unwatermarked control", "///"),
            ):
                values = branch_data[branch]
                bars = ax.bar(np.array([0, 1]) + shift, values["percent"], width=0.33,
                              color=colors[branch], edgecolor="#ffffff" if not hatch else "#8c3500",
                              hatch=hatch, linewidth=0.65, alpha=0.88, label=label, zorder=3)
                if row == col == 0:
                    legend_handles.append(bars)
                for bar, count, percent in zip(bars, values["counts"], values["percent"]):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0,
                            f"{percent:.2f}%\n{count}/{values['n']}",
                            ha="center", va="bottom", fontsize=7.8, linespacing=1.25, color="#242424")
            wm, control = branch_data["watermarked"], branch_data["unwatermarked"]
            ax.text(0.02, 1.055, f"WM: {wm['mean']:.3f} \u00b1 {wm['std']:.3f}",
                    transform=ax.transAxes, color=colors["watermarked"], fontsize=8, va="bottom")
            ax.text(0.53, 1.055, f"Control: {control['mean']:.3f} \u00b1 {control['std']:.3f}",
                    transform=ax.transAxes, color=colors["unwatermarked"], fontsize=8, va="bottom")
            ax.set(xlim=(-0.62, 1.62), ylim=(0, 110), xlabel="z-score (discrete values)")
            ax.set_xticks([0, 1], ["\u22120.3333", "3.0000"])
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=3)
            if col == 0:
                ax.set_ylabel("Share within each class (%)")
    fig.legend(handles=legend_handles, labels=["Watermarked (WM)", "Unwatermarked control"],
               loc="upper center", bbox_to_anchor=(0.5, 0.918), ncol=2, frameon=False, fontsize=9)
    fig.text(0.105, 0.079,
             "Bars: exact empirical mass at each score; labels: percentage and count / scoreable n.\n"
             "Panel summaries: mean \u00b1 std across tasks (ddof = 0). No density smoothing.\n"
             "Each class is normalized separately; 7 unscoreable MBPP controls are excluded. Seed 42.",
             fontsize=7.8, va="top", linespacing=1.45, color="#444444")
    artifacts = []
    for suffix in ("png", "pdf"):
        output = LEDGER / f"seqmark_z_distribution.{suffix}"
        fig.savefig(output, dpi=300)
        artifacts.append(output)
    plt.close(fig)
    csv_path = LEDGER / "seqmark_z_distribution.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    artifacts.append(csv_path)
    PUBLISH.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        shutil.copy2(artifact, PUBLISH / artifact.name)
    save(LEDGER / "metrics.json", metrics)
    save(LEDGER / "diagnostics.json", {
        "source_hashes_match": True, "unique_scores_per_class": 2, "class_count": 8,
        "counts_match_denominators": True, "masses_sum_to_one": True,
        "mean_and_std_reconstructed_from_masses": True,
    })
    manifest.update(status="completed", artifacts=[
        {"path": str(p), "sha256": sha(p), "published_path": str(PUBLISH / p.name)} for p in artifacts
    ])
    print(json.dumps({"artifacts": manifest["artifacts"], "class_summaries": metrics}, indent=2))
except BaseException as error:
    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    manifest["ended_at"] = now()
    save(manifest_path, manifest)
