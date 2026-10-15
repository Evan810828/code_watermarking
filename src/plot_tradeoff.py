"""Detection rate vs performance degradation, one panel per dataset.

Reads the unified-protocol numbers straight out of `out/README.md` so the figure
can never drift from the table it illustrates.

  x = Δpass@1 (unwatermarked − watermarked), i.e. the quality the watermark costs
  y = TPR@5%FPR against the paired unwatermarked completions

Top-LEFT is the good corner: detectable and cheap. Official implementations and
LiveCodeBench are excluded per the figure's scope.

    python plot_tradeoff.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "figures"

# dataviz categorical slots 1/2/3/7, validated all-pairs on the light surface
# (worst CVD ΔE 9.2, worst normal-vision ΔE 16.3). Slot 4 (yellow) is skipped:
# it fails the all-pairs floors against orange.
COLOR = {"KGW": "#2a78d6", "SWEET": "#eb6834", "SemStamp": "#1baf7a", "CodeWM": "#4a3aa7"}
INK, INK_MUTED, GRID, SURFACE = "#141413", "#6b6a63", "#e6e4dd", "#fcfcfb"
DATASETS = [("humaneval", "HumanEval"), ("mbpp", "MBPP")]

ROW = re.compile(
    r"\| (KGW|SWEET|SemStamp|CodeWM) \| ([^|]+?) \| (\w+) \| (\d+) \| (\d+) \| "
    r"(\d+) \(\d+%\) \| ([\d.]+) \| [\d.]+% \| ([\d.]+)% \| [+-][\d.]+ \| [+-][\d.]+ \| "
    r"([\d.]+)% \| ([\d.]+)% \|"
)


def load(readme: Path) -> list[dict]:
    rows = []
    for line in readme.read_text().splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        method, impl, ds = m.group(1), m.group(2).strip(), m.group(3)
        if "official" in impl or ds not in dict(DATASETS):
            continue
        n, unsc = int(m.group(4)), int(m.group(6))
        rows.append({
            "method": method, "impl": impl, "ds": ds,
            "tpr5": float(m.group(8)),
            "dpass": float(m.group(10)) - float(m.group(9)),
            "unsc_pct": 100 * unsc / n,
            # δ is the only thing distinguishing CodeWM's two operating points
            "delta": (re.search(r"δ=(\d+)", impl) or [None, None])[1],
        })
    return rows


def label_for(r: dict) -> str:
    name = r["method"] + (f" δ={r['delta']}" if r["delta"] else "")
    # an unscoreable fraction is not a footnote: the TPR beside it was computed
    # on the remainder, so it belongs in the label itself
    return name + (f"\n({r['unsc_pct']:.0f}% unscoreable)" if r["unsc_pct"] >= 5 else "")


def place_labels(rows: list[dict], xlim, ylim) -> dict[int, tuple[int, int, str, str]]:
    """Pick a label offset per point that does not land on another point.

    SemStamp and CodeWM δ=2 sit almost on top of each other on both datasets —
    that proximity IS the finding (both are near-free and near-undetectable), so
    the fix is to place labels around the cluster rather than to move the marks.
    Greedy: try the four diagonals, keep the first whose text box is clear.
    """
    xs, ys = xlim[1] - xlim[0], ylim[1] - ylim[0]
    norm = [((r["dpass"] - xlim[0]) / xs, (r["tpr5"] - ylim[0]) / ys) for r in rows]
    # (dx, dy) in points; near offsets first, then far ones that need a leader
    CAND = [(12, 9), (-12, 9), (12, -11), (-12, -11),
            (0, 24), (0, -26), (46, 20), (-46, 20), (46, -22), (-46, -22),
            (64, 4), (-64, 4)]
    out, taken = {}, []
    order = sorted(range(len(rows)), key=lambda i: -norm[i][1])  # top-down
    for i in order:
        px, py = norm[i]
        best = None
        for dx, dy in CAND:
            # convert the offset to normalised units (fig is ~6.4in x ~4in per panel)
            bx = px + dx / 430.0
            by = py + dy / 300.0
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "bottom" if dy > 0 else "top"
            clash = any(abs(bx - ox) < 0.12 and abs(by - oy) < 0.06
                        for j, (ox, oy) in enumerate(norm) if j != i)
            clash = clash or any(abs(bx - tx) < 0.24 and abs(by - ty) < 0.075
                                 for tx, ty in taken)
            inside = 0.03 < bx < 0.97 and 0.03 < by < 0.97
            if inside and not clash:
                best = (dx, dy, ha, va, bx, by)
                break
        if best is None:
            best = (12, 9, "left", "bottom", px + 0.028, py + 0.03)
        out[i] = best[:4]
        taken.append(best[4:])
    return out


def draw(ax, rows: list[dict], title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

    # the δ sweep is a trajectory of one method, not two unrelated points
    sweep = sorted([r for r in rows if r["delta"]], key=lambda r: float(r["delta"]))
    if len(sweep) > 1:
        ax.plot([r["dpass"] for r in sweep], [r["tpr5"] for r in sweep],
                color=COLOR["CodeWM"], linewidth=2, alpha=0.35, zorder=1)

    ax.set_ylim(-16, 116)
    xs = [r["dpass"] for r in rows]
    ax.set_xlim(min(xs) - 10, max(xs) + 22)

    for r in rows:
        ax.scatter(r["dpass"], r["tpr5"], s=150, color=COLOR[r["method"]], zorder=3,
                   edgecolor=SURFACE, linewidth=2)
    for i, (dx, dy, ha, va) in place_labels(rows, ax.get_xlim(), ax.get_ylim()).items():
        # a label pushed clear of a crowded cluster needs a leader back to its mark
        far = abs(dx) > 20 or abs(dy) > 20
        ax.annotate(
            label_for(rows[i]), (rows[i]["dpass"], rows[i]["tpr5"]),
            textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
            fontsize=9, color=INK, linespacing=1.35, zorder=4,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8,
                            shrinkA=2, shrinkB=6) if far else None,
        )

    ax.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    ax.set_xlabel("Δ pass@1  (unwatermarked − watermarked, pp)", fontsize=10, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def main() -> int:
    rows = load(ROOT / "out" / "README.md")
    if not rows:
        print("no rows parsed from out/README.md — has the table format changed?")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, DATASETS):
        draw(ax, [r for r in rows if r["ds"] == key], title)
    axes[0].set_ylabel("TPR @ 5% FPR  (%)", fontsize=10, color=INK_MUTED)

    fig.suptitle("Detection vs quality cost — Qwen3.5-4B, pilot implementations",
                 fontsize=14, color=INK, x=0.005, ha="left", y=0.99)
    fig.text(0.005, 0.915,
             "Top-left is better: detectable at little quality cost.  Detection is "
             "TPR at 5% FPR against the paired unwatermarked completions;\nquality "
             "cost is the pass@1 drop.  Official implementations and LiveCodeBench "
             "excluded.",
             fontsize=9.5, color=INK_MUTED, ha="left", va="top", linespacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.87])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tradeoff_qwen3.5-4b.{ext}", dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}/tradeoff_qwen3.5-4b.{{png,pdf}}  ({len(rows)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
