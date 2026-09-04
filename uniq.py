"""Solution counting (up to a limit) and unique-puzzle generation."""
import numpy as np


def count_solutions(grid, limit=2):
    """Number of completions of a (81,) grid with 0 = hole, capped at `limit`."""
    g = [int(x) for x in grid]; rows = [0] * 9; cols = [0] * 9; boxes = [0] * 9
    for i, d in enumerate(g):
        if d:
            r, c = divmod(i, 9); b = (r // 3) * 3 + c // 3; bit = 1 << d
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
    empties = [i for i in range(81) if g[i] == 0]
    n = [0]

    def rec():
        if n[0] >= limit:
            return
        best, bestc = None, None
        for i in empties:
            if g[i]:
                continue
            r, c = divmod(i, 9); b = (r // 3) * 3 + c // 3; used = rows[r] | cols[c] | boxes[b]
            cands = [d for d in range(1, 10) if not (used >> d) & 1]
            if best is None or len(cands) < len(bestc):
                best, bestc = i, cands
                if not cands:
                    return
        if best is None:
            n[0] += 1; return
        r, c = divmod(best, 9); b = (r // 3) * 3 + c // 3
        for d in bestc:
            bit = 1 << d; g[best] = d; rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
            rec()
            g[best] = 0; rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit
            if n[0] >= limit:
                return
    rec()
    return n[0]


def make_unique_puzzle(solved, n_holes, rng):
    """Greedily remove cells in random order, keeping the solution unique.
    Returns (holed, achieved_holes); achieved may be < n_holes if it gets stuck."""
    holed = np.array(solved, dtype=np.uint8).copy()
    order = rng.permutation(81)
    holes = 0
    for i in order:
        if holes >= n_holes:
            break
        v = holed[i]; holed[i] = 0
        if count_solutions(holed) == 1:
            holes += 1
        else:
            holed[i] = v
    return holed, holes
