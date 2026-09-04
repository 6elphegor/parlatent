"""Held-out accuracy curves for several runs: small multiples per hole level.
Top row: solve rate (valid grid consistent with givens).  Bottom row: hole-cell accuracy.
Usage: python plot_acc.py loop4_lr0.01 deep16_lr0.01 [--out plots/acc_best.png]
       python plot_acc.py loop4_lr0.01 deep16_lr0.01 --levels 40 --metrics solve --out plots/solve40_best.png"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
def load(name):
    steps, solve, acc = [], {}, {}
    for line in open(f"runs/{name}/metrics.jsonl"):
        d = json.loads(line)
        if "eval" in d:
            steps.append(d["step"])
            for h in d["eval"]:
                solve.setdefault(h, []); acc.setdefault(h, [])
                solve[h].append(d["eval"][h]["solve_rate"][-1][-1])
                acc[h].append(d["eval"][h]["hole_acc"][-1][-1])
    return np.array(steps), solve, acc


def label_for(name):
    cfg = json.load(open(f"runs/{name}/config.json"))
    arch = f"{cfg['n_layers']}L x {cfg['n_loops']} loops" if cfg["n_loops"] > 1 else f"{cfg['n_layers']}L unrolled"
    return f"{name}  ({arch}, Muon lr {cfg['muon_lr']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="plots/acc_best.png")
    ap.add_argument("--levels", default="20,30,40,50", help="hole levels, comma-separated")
    ap.add_argument("--metrics", default="solve,acc", help="rows: solve and/or acc")
    a = ap.parse_args()
    data = {n: load(n) for n in a.runs}
    LEVELS = a.levels.split(",")
    rows = [m for m in [("solve rate", 1), ("hole-cell accuracy", 2)]
            if ("solve" if m[1] == 1 else "acc") in a.metrics.split(",")]
    single = len(rows) == 1 and len(LEVELS) == 1
    figsize = (7, 4.5) if single else (3 * len(LEVELS), 3 * len(rows))
    fig, axes = plt.subplots(len(rows), len(LEVELS), figsize=figsize, dpi=150, facecolor=SURFACE,
                             sharex=True, squeeze=False)
    for r, (title, idx) in enumerate(rows):
        for c, h in enumerate(LEVELS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ys = {n: np.array((data[n][1] if idx == 1 else data[n][2])[h]) for n in a.runs}
            rank = {n: k for k, n in enumerate(sorted(a.runs, key=lambda n: ys[n][-1]))}
            for i, name in enumerate(a.runs):
                steps, y = data[name][0], ys[name]
                ax.plot(steps, y, color=SERIES[i], lw=2, marker="o", ms=3.5, mec=SURFACE, mew=0.8,
                        label=label_for(name) if (r == 0 and c == 0) else None)
                # spread end labels apart when the two final values are close
                spread = max(abs(ys[m][-1] - y[-1]) for m in a.runs if m != name) < 0.06 if len(a.runs) > 1 else False
                dy = (-5 + 10 * rank[name]) if spread else 0
                ax.annotate(f"{y[-1]:.0%}", (steps[-1], y[-1]), xytext=(4, dy), textcoords="offset points",
                            va="center", fontsize=8, color=TEXT2)
            if r == 0 and not single:
                ax.set_title(f"{h} holes", color=TEXT, fontsize=11, loc="left")
            if c == 0:
                ax.set_ylabel(title, color=TEXT2)
            if r == len(rows) - 1:
                ax.set_xlabel("training step", color=TEXT2)
            lo = 0.0 if idx == 1 else 0.5
            ax.set_ylim(lo, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_xlim(0, max(data[n][0][-1] for n in a.runs) * 1.15)
            ax.grid(True, color=GRID, lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=TEXT2, labelsize=8.5)
    if single:
        fig.suptitle(f"Held-out {rows[0][0]} at {LEVELS[0]} holes (final loop, last copy)",
                     x=0.01, y=0.985, ha="left", color=TEXT, fontsize=12)
        axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.suptitle("Held-out accuracy by hole count, best learning rate per architecture (final loop, last copy)",
                     x=0.01, y=0.985, ha="left", color=TEXT, fontsize=12)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper left",
                   bbox_to_anchor=(0.005, 0.955), ncol=2, labelcolor=TEXT)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(a.out, facecolor=SURFACE)
    print("saved", a.out)


if __name__ == "__main__":
    main()
