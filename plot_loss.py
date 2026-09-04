"""Training-loss curves for the best run of each architecture on one axis.
Usage: python plot_loss.py loop4_lr0.01 deep16_lr0.01 [--out plots/loss_best.png]"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load(name):
    steps, train, ev_steps, ev = [], [], [], []
    for line in open(f"runs/{name}/metrics.jsonl"):
        d = json.loads(line)
        if "train_loss" in d:
            steps.append(d["step"]); train.append(d["loop_losses"][-1])   # final-loop loss
        elif "eval" in d:
            ev_steps.append(d["step"]); ev.append(d["eval"]["40"]["loop_losses"][-1])
    return np.array(steps), np.array(train), np.array(ev_steps), np.array(ev)


def smooth(x, w=5):
    """Centered rolling mean (w log points = w*50 steps); no lag."""
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def label_for(name):
    cfg = json.load(open(f"runs/{name}/config.json"))
    arch = f"{cfg['n_layers']}L x {cfg['n_loops']} loops" if cfg["n_loops"] > 1 else f"{cfg['n_layers']}L unrolled"
    return f"{name}  ({arch}, Muon lr {cfg['muon_lr']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="plots/loss_best.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    data = {n: load(n) for n in a.runs}
    finals = {n: smooth(data[n][1])[-1] for n in a.runs}
    rank = {n: r for r, n in enumerate(sorted(a.runs, key=lambda n: finals[n]))}   # 0 = lowest
    for i, name in enumerate(a.runs):
        c = SERIES[i]
        s, tr, es, ev = data[name]
        ax.plot(s, tr, color=c, lw=1, alpha=0.25)                      # raw
        sm = smooth(tr)
        ax.plot(s, sm, color=c, lw=2, label=label_for(name))            # smoothed
        ax.plot(es[1:], ev[1:], "o", color=c, ms=5, mec=SURFACE, mew=1)  # held-out eval, 40 holes
        ax.annotate(f"{sm[-1]:.3f}", (s[-1], sm[-1]), xytext=(6, -6 + 12 * rank[name]),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT2)
    ax.set_yscale("log")
    ax.set_xlabel("training step", color=TEXT2)
    ax.set_ylabel("next-token loss (final loop)", color=TEXT2)
    ax.set_title("Training loss, best learning rate per architecture", color=TEXT, loc="left",
                 fontsize=12, pad=22)
    ax.text(0, 1.02, "lines: train loss (light = raw, bold = 250-step mean);  dots: held-out eval loss at 40 holes",
            transform=ax.transAxes, fontsize=8.5, color=TEXT2, va="bottom")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(False, which="minor")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.set_xlim(0, max(data[n][0][-1] for n in a.runs) * 1.08)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, facecolor=SURFACE)
    print("saved", a.out)


if __name__ == "__main__":
    main()
