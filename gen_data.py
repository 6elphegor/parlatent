"""Generate random *solved* sudoku grids by randomized backtracking.

Why not the "permute a base pattern" trick: every grid in that family shares
the same box-row / box-column triple structure, which a transformer can learn
as a shortcut. Randomized backtracking samples much closer to uniform over
all valid grids.  Grids are saved as uint8 (N, 81) with digits 1..9.
Symmetry augmentation (band/stack/row/col/digit perms, transpose) is applied
on the fly at training time in sudoku.py, so N seeds -> ~1e12 * N grids.
"""
import argparse, os, time, random
import numpy as np
from multiprocessing import Pool


def random_solved_grid(rng: random.Random) -> list:
    grid = [0] * 81
    rows = [0] * 9   # bitmasks of used digits
    cols = [0] * 9
    boxes = [0] * 9

    def rec(i):
        if i == 81:
            return True
        r, c = divmod(i, 9)
        b = (r // 3) * 3 + c // 3
        used = rows[r] | cols[c] | boxes[b]
        cands = [d for d in range(1, 10) if not (used >> d) & 1]
        rng.shuffle(cands)
        for d in cands:
            bit = 1 << d
            grid[i] = d
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
            if rec(i + 1):
                return True
            rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit
        grid[i] = 0
        return False

    assert rec(0)
    return grid


def worker(args):
    seed, n = args
    rng = random.Random(seed)
    out = np.empty((n, 81), dtype=np.uint8)
    for i in range(n):
        out[i] = random_solved_grid(rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--out", default="data/solved_grids.npy")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    chunk = 2000
    jobs = [(a.seed * 1_000_003 + k, min(chunk, a.n - k * chunk))
            for k in range((a.n + chunk - 1) // chunk)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        parts = []
        for j, part in enumerate(pool.imap(worker, jobs)):
            parts.append(part)
            if (j + 1) % 10 == 0 or j + 1 == len(jobs):
                done = sum(len(p) for p in parts)
                print(f"{done}/{a.n} grids  {time.time()-t0:.1f}s", flush=True)
    grids = np.concatenate(parts)[: a.n]
    # sanity check validity
    from sudoku import is_valid_grid
    assert is_valid_grid(grids).all()
    np.save(a.out, grids)
    print(f"saved {grids.shape} -> {a.out}")


if __name__ == "__main__":
    main()
