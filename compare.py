"""Compare final evals of several runs: python compare.py loop4 noloop"""
import json, sys


def last_eval(name):
    ev = None
    for line in open(f"runs/{name}/metrics.jsonl"):
        d = json.loads(line)
        if "eval" in d:
            ev = d
    return ev


def main():
    names = sys.argv[1:] or ["loop4", "noloop"]
    evs = {n: last_eval(n) for n in names}
    levels = list(next(iter(evs.values()))["eval"].keys())
    print("final-loop, last-copy solve rate (valid grid consistent with givens)")
    print(f"{'holes':>6} " + " ".join(f"{n:>10}" for n in names))
    for h in levels:
        print(f"{h:>6} " + " ".join(f"{evs[n]['eval'][h]['solve_rate'][-1][-1]:>10.3f}" for n in names))
    print("\nfinal-loop, last-copy hole-cell accuracy")
    print(f"{'holes':>6} " + " ".join(f"{n:>10}" for n in names))
    for h in levels:
        print(f"{h:>6} " + " ".join(f"{evs[n]['eval'][h]['hole_acc'][-1][-1]:>10.3f}" for n in names))
    for n in names:
        print(f"\n{n} (step {evs[n]['step']}): solve_rate[loop][copy] per hole level")
        for h in levels:
            m = evs[n]["eval"][h]["solve_rate"]
            print(f"  holes={h:>2}: " + " | ".join(" ".join(f"{v:.3f}" for v in row) for row in m))


def sweep_summary(prefixes=("loop4_lr", "deep16_lr"), metric_holes=("30", "40")):
    """Best Muon LR per architecture by final-loop last-copy solve rate."""
    import glob, os
    for pre in prefixes:
        rows = []
        for path in sorted(glob.glob(f"runs/{pre}*/metrics.jsonl")):
            name = os.path.basename(os.path.dirname(path))
            ev = last_eval(name)
            if ev is None:
                continue
            sr = {h: ev["eval"][h]["solve_rate"][-1][-1] for h in ev["eval"]}
            rows.append((float(name[len(pre):]), ev["step"], sr))
        rows.sort()
        print(f"\n{pre}*  (step, solve rate per hole level)")
        for lr, step, sr in rows:
            print(f"  lr={lr:<6} step={step:<5} " + " ".join(f"{h}:{sr[h]:.3f}" for h in sr))
        if rows:
            best = max(rows, key=lambda r: sum(r[2][h] for h in metric_holes))
            print(f"  best (by solve@{'+'.join(metric_holes)}): lr={best[0]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        sweep_summary()
    else:
        main()
