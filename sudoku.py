"""Sudoku tokenization, symmetry augmentation, hole punching, validity checks."""
import numpy as np
import torch

# ---- vocabulary -----------------------------------------------------------
START_HOLED, END_HOLED = 0, 1
DIGIT_OFFSET = 1            # digit d (1..9) -> token d + 1  == 2..10
HOLE, ROW_SEP = 11, 12
START_SOLVED, END_SOLVED = 13, 14
VOCAB_SIZE = 15
TOKEN_NAMES = (["<holed>", "</holed>"] + [str(d) for d in range(1, 10)]
               + ["_", "|", "<solved>", "</solved>"])

GRID_TOKENS = 81 + 8          # 9 rows of 9 digits with row_sep between rows
SEG_LEN = GRID_TOKENS + 2     # + start/end delimiters

# positions (within a 91-token segment) of the 81 cells and of the 8 separators
_cell_pos = []
_sep_pos = []
p = 1
for r in range(9):
    for c in range(9):
        _cell_pos.append(p); p += 1
    if r < 8:
        _sep_pos.append(p); p += 1
assert p == SEG_LEN - 1
CELL_POS = np.array(_cell_pos)
SEP_POS = np.array(_sep_pos)


def seq_len(n_repeats: int) -> int:
    return SEG_LEN * (1 + n_repeats)


def grids_to_segment(grids: np.ndarray, start_tok: int, end_tok: int) -> np.ndarray:
    """grids: (B, 81) with values 1..9 or 0 (=hole) -> (B, SEG_LEN) tokens."""
    B = grids.shape[0]
    seg = np.empty((B, SEG_LEN), dtype=np.int64)
    seg[:, 0] = start_tok
    seg[:, -1] = end_tok
    seg[:, SEP_POS] = ROW_SEP
    cells = np.where(grids == 0, HOLE, grids + DIGIT_OFFSET)
    seg[:, CELL_POS] = cells
    return seg


def segment_to_grids(seg: np.ndarray) -> np.ndarray:
    """(B, SEG_LEN) tokens -> (B, 81) digits (values outside 1..9 -> 0)."""
    cells = seg[:, CELL_POS] - DIGIT_OFFSET
    return np.where((cells >= 1) & (cells <= 9), cells, 0).astype(np.uint8)


def build_sequences(holed: np.ndarray, solved: np.ndarray, n_repeats: int):
    """Returns (inp, tgt), both (B, T).
    inp = [holed segment] + n_repeats * [solved segment with every cell = HOLE]
    tgt = [holed segment] + n_repeats * [true solved segment]
    Standard next-token shift: logits[:, i] predicts tgt[:, i+1].  So the model also
    predicts the next given of the holed puzzle (auxiliary signal), and at a blank
    cell of a solved copy it predicts the *next* cell's digit."""
    B = holed.shape[0]
    holed_seg = grids_to_segment(holed, START_HOLED, END_HOLED)
    blank_seg = grids_to_segment(np.zeros((B, 81), dtype=holed.dtype), START_SOLVED, END_SOLVED)
    sol_seg = grids_to_segment(solved, START_SOLVED, END_SOLVED)
    inp = np.concatenate([holed_seg] + [blank_seg] * n_repeats, axis=1)
    tgt = np.concatenate([holed_seg] + [sol_seg] * n_repeats, axis=1)
    return inp, tgt


# ---- validity -------------------------------------------------------------
def is_valid_grid(grids: np.ndarray) -> np.ndarray:
    """(B, 81) -> (B,) bool: every row / col / box is a permutation of 1..9."""
    g = grids.reshape(-1, 9, 9).astype(np.int64)
    ok = ((g >= 1) & (g <= 9)).reshape(len(g), -1).all(1)
    onehot = np.zeros((len(g), 9, 9, 10), dtype=np.int64)
    np.put_along_axis(onehot, g[..., None], 1, axis=-1)
    onehot = onehot[..., 1:]                                   # (B,9,9,9)
    rows_ok = (onehot.sum(2) == 1).all((1, 2))
    cols_ok = (onehot.sum(1) == 1).all((1, 2))
    boxes = onehot.reshape(len(g), 3, 3, 3, 3, 9).sum((2, 4))  # (B,3,3,9)
    boxes_ok = (boxes == 1).all((1, 2, 3))
    return ok & rows_ok & cols_ok & boxes_ok


def is_consistent(holed: np.ndarray, pred: np.ndarray) -> np.ndarray:
    given = holed != 0
    return ((pred == holed) | ~given).all(1)


# ---- symmetry augmentation ------------------------------------------------
def augment(grids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random sudoku symmetry: band perm, row-in-band perms, stack perm,
    col-in-stack perms, digit relabel, optional transpose.  Vectorized."""
    B = grids.shape[0]
    g = grids.reshape(B, 9, 9)

    def line_perm():
        band = rng.permuted(np.tile(np.arange(3), (B, 1)), axis=1)         # (B,3)
        within = rng.permuted(np.tile(np.arange(3), (B, 3, 1)), axis=2)    # (B,3,3)
        return (band[:, :, None] * 3 + within).reshape(B, 9)               # (B,9)

    rp, cp = line_perm(), line_perm()
    bi = np.arange(B)[:, None, None]
    g = g[bi, rp[:, :, None], cp[:, None, :]]
    digit_map = np.zeros((B, 10), dtype=grids.dtype)
    digit_map[:, 1:] = rng.permuted(np.tile(np.arange(1, 10), (B, 1)), axis=1)
    g = np.take_along_axis(digit_map, g.reshape(B, 81).astype(np.int64), axis=1).reshape(B, 9, 9)
    t = rng.random(B) < 0.5
    g = np.where(t[:, None, None], g.transpose(0, 2, 1), g)
    return g.reshape(B, 81)


def punch_holes(solved: np.ndarray, n_holes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Zero out n_holes[b] random cells of each grid."""
    B = solved.shape[0]
    rank = np.argsort(rng.random((B, 81)), axis=1).argsort(axis=1)
    mask = rank < n_holes[:, None]
    return np.where(mask, 0, solved)


class SudokuData:
    """Samples training / eval batches from a bank of solved seed grids."""

    def __init__(self, path: str, n_repeats: int, min_holes: int, max_holes: int,
                 n_val: int = 10_000, seed: int = 0):
        grids = np.load(path)
        assert grids.shape[1] == 81
        self.train = grids[:-n_val]
        self.val = grids[-n_val:]
        self.n_repeats = n_repeats
        self.min_holes, self.max_holes = min_holes, max_holes
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, split: str = "train", n_holes=None,
               rng: np.random.Generator | None = None):
        rng = rng or self.rng
        bank = self.train if split == "train" else self.val
        idx = rng.integers(0, len(bank), batch_size)
        solved = augment(bank[idx], rng)
        if n_holes is None:
            n_holes = rng.integers(self.min_holes, self.max_holes + 1, batch_size)
        else:
            n_holes = np.full(batch_size, n_holes)
        holed = punch_holes(solved, n_holes, rng)
        inp, tgt = build_sequences(holed, solved, self.n_repeats)
        return dict(inp=torch.from_numpy(inp), tgt=torch.from_numpy(tgt),
                    holed=holed, solved=solved, n_holes=n_holes)


def loss_mask(n_repeats: int) -> torch.Tensor:
    """(T-1,) bool over shifted target positions (target index j -> row j-1).
    Loss on every next-token prediction: the holed puzzle's givens, structural
    tokens, and the solved copies' cells."""
    return torch.ones(seq_len(n_repeats) - 1, dtype=torch.bool)


def hole_mask(n_repeats: int, holed: np.ndarray) -> np.ndarray:
    """(B, T-1) bool over *shifted* target positions: the hole cells of each solved
    copy.  Given cells only need copying, so accuracy is reported on holes."""
    B = holed.shape[0]
    m = np.zeros((B, seq_len(n_repeats) - 1), dtype=bool)
    for k in range(n_repeats):
        m[:, SEG_LEN * (1 + k) + CELL_POS - 1] = holed == 0
    return m


def copy_slices(n_repeats: int):
    """Token index ranges [start, end) of each solved copy.  In the shifted
    (prediction) frame use [start-1, end-1)."""
    return [(SEG_LEN * (1 + k), SEG_LEN * (2 + k)) for k in range(n_repeats)]


def decode(seq) -> str:
    return " ".join(TOKEN_NAMES[int(t)] for t in seq)
