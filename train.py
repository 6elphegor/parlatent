"""Train a looped transformer to solve sudoku.

Input:    [<holed> 81 cells </holed>] + n_repeats x [<solved> 81 x HOLE </solved>]
Target:   [<holed> 81 cells </holed>] + n_repeats x [<solved> true solution </solved>]
Loss:     next-token CE (shifted by one) on all positions, averaged over loops.  The
          holed segment adds an auxiliary "predict the next given" signal; the blank
          solved slots never contain the solution, so the model cannot copy it.
Infer:    a single forward pass; copy k's grid = argmax at the positions preceding
          copy k's cells.
Loops:    n_loops defaults to n_repeats (per the spec); pass --n_loops 1 for the
          no-looping ablation (same data, same params, single pass).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

import sudoku as S
from model import Config, LoopedTransformer
from muon import Muon


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/solved_grids.npy")
    ap.add_argument("--n_repeats", type=int, default=4, help="solved copies per sequence")
    ap.add_argument("--n_loops", type=int, default=None, help="default = n_repeats; 1 = no looping")
    ap.add_argument("--min_holes", type=int, default=0)
    ap.add_argument("--max_holes", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--ffn_mult", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--eval_holes", default="10,20,30,40,50")
    ap.add_argument("--eval_n", type=int, default=512, help="puzzles per hole level")
    ap.add_argument("--no_causal", action="store_true", help="bidirectional attention")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs")
    a = ap.parse_args()
    if a.n_loops is None:
        a.n_loops = a.n_repeats
    return a


def lr_mult(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, data, args, dev, rng):
    """Single forward pass per batch.  For every (loop, copy): loss, accuracy on hole
    cells, solve rate (valid sudoku consistent with givens) and exact match."""
    model.eval()
    levels = [int(h) for h in args.eval_holes.split(",")]
    mask = S.loss_mask(args.n_repeats).to(dev)
    out = {}
    for h in levels:
        b = data.sample(args.eval_n, "val", n_holes=h, rng=rng)
        inp, tgt = b["inp"].to(dev), b["tgt"][:, 1:].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, losses, all_logits = model(inp, tgt, mask, return_all=True)
        hm = S.hole_mask(args.n_repeats, b["holed"])
        L, K = args.n_loops, args.n_repeats
        acc, valid, exact = np.zeros((L, K)), np.zeros((L, K)), np.zeros((L, K))
        for l, lg in enumerate(all_logits):
            pred = lg[:, :-1].argmax(-1).cpu().numpy()          # shifted frame
            for k, (s, e) in enumerate(S.copy_slices(K)):
                g = S.segment_to_grids(pred[:, s - 1:e - 1])
                hole = hm[:, s - 1:e - 1][:, S.CELL_POS]
                acc[l, k] = ((g == b["solved"]) & hole).sum() / max(1, hole.sum())
                valid[l, k] = (S.is_valid_grid(g) & S.is_consistent(b["holed"], g)).mean()
                exact[l, k] = (g == b["solved"]).all(1).mean()
        out[h] = dict(loss=loss.item(), loop_losses=[x.item() for x in losses],
                      hole_acc=acc.tolist(), solve_rate=valid.tolist(), exact=exact.tolist())
    model.train()
    return out


def fmt_matrix(m):
    return " | ".join(" ".join(f"{v:.3f}" for v in row) for row in m)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    metrics_f = open(os.path.join(run_dir, "metrics.jsonl"), "a")

    data = S.SudokuData(args.data, args.n_repeats, args.min_holes, args.max_holes, seed=args.seed)
    T = S.seq_len(args.n_repeats)
    cfg = Config(vocab_size=S.VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
                 n_layers=args.n_layers, ffn_mult=args.ffn_mult, n_loops=args.n_loops,
                 max_seq_len=T, causal=not args.no_causal)
    model = LoopedTransformer(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"run={args.name} params={n_params/1e6:.2f}M seq_len={T} n_repeats={args.n_repeats} "
          f"n_loops={args.n_loops} causal={cfg.causal} holes=[{args.min_holes},{args.max_holes}]", flush=True)

    muon_p, adam_p = model.param_groups()
    opt_muon = Muon(muon_p, lr=args.muon_lr, momentum=0.95, weight_decay=args.weight_decay)
    opt_adam = torch.optim.AdamW(adam_p, lr=args.adam_lr, betas=(0.9, 0.95),
                                 weight_decay=args.weight_decay)
    opts = [opt_muon, opt_adam]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    fwd = torch.compile(model) if args.compile else model
    mask = S.loss_mask(args.n_repeats).to(dev)
    eval_rng = np.random.default_rng(1234)   # fixed eval puzzles across evals/runs

    t0 = time.time()
    tok_count = 0
    for step in range(args.steps + 1):
        last = step == args.steps
        if step % args.eval_interval == 0 or last:
            ev = evaluate(model, data, args, dev, np.random.default_rng(1234))
            print(f"--- eval step {step}   (matrices are [loop][copy])")
            for h in ev:
                print(f"  holes={h:2d} loss={ev[h]['loss']:.4f} "
                      f"loop_losses={[round(x, 4) for x in ev[h]['loop_losses']]}")
                print(f"           hole_acc:   {fmt_matrix(ev[h]['hole_acc'])}")
                print(f"           solve_rate: {fmt_matrix(ev[h]['solve_rate'])}")
                print(f"           exact:      {fmt_matrix(ev[h]['exact'])}")
            final = {h: ev[h]['solve_rate'][-1][-1] for h in ev}
            print(f"  final-loop last-copy solve_rate: {final}", flush=True)
            metrics_f.write(json.dumps(dict(step=step, eval=ev)) + "\n"); metrics_f.flush()
            torch.save(dict(model=model.state_dict(), cfg=cfg.__dict__, args=vars(args), step=step),
                       os.path.join(run_dir, "ckpt.pt"))
            if last:
                break

        m = lr_mult(step, args.steps, args.warmup)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * m

        batch = data.sample(args.batch_size, "train")
        inp = batch["inp"].to(dev, non_blocking=True)
        tgt = batch["tgt"][:, 1:].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, losses, _ = fwd(inp, tgt, mask)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for o in opts:
            o.step()
        model.zero_grad(set_to_none=True)
        tok_count += inp.numel()

        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:6d} loss {loss.item():.4f} loops={[round(x.item(), 4) for x in losses]} "
                  f"gn {gn.item():.2f} lr x{m:.3f} {tok_count/dt/1e3:.1f}k tok/s {dt:.0f}s", flush=True)
            metrics_f.write(json.dumps(dict(step=step, train_loss=loss.item(),
                                            loop_losses=[x.item() for x in losses])) + "\n")


if __name__ == "__main__":
    main()
