"""D4 dihedral group: 8 symmetry transforms for 15×15 board positions.

Transforms: identity, 3 rotations (90/180/270), horizontal flip,
vertical flip, main-diagonal flip, anti-diagonal flip.
"""

from __future__ import annotations

import numpy as np

from .board import BOARD_SIZE, NUM_CELLS


def _build_transforms() -> list[np.ndarray]:
    """Pre-compute 8 permutation tables, each mapping pos → transformed pos."""
    transforms: list[np.ndarray] = []
    coords = np.array([(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)])
    r_arr, c_arr = coords[:, 0], coords[:, 1]
    n = BOARD_SIZE - 1

    def _to_perm(r_out: np.ndarray, c_out: np.ndarray) -> np.ndarray:
        return (r_out * BOARD_SIZE + c_out).astype(np.int64)

    transforms.append(_to_perm(r_arr, c_arr))                        # identity
    transforms.append(_to_perm(c_arr, n - r_arr))                    # rot 90 CW
    transforms.append(_to_perm(n - r_arr, n - c_arr))                # rot 180
    transforms.append(_to_perm(n - c_arr, r_arr))                    # rot 270 CW
    transforms.append(_to_perm(r_arr, n - c_arr))                    # flip horizontal
    transforms.append(_to_perm(n - r_arr, c_arr))                    # flip vertical
    transforms.append(_to_perm(c_arr, r_arr))                        # flip main diag
    transforms.append(_to_perm(n - c_arr, n - r_arr))                # flip anti diag
    return transforms


TRANSFORMS = _build_transforms()
NUM_TRANSFORMS = len(TRANSFORMS)


def transform_pos(pos: int, t: int) -> int:
    return int(TRANSFORMS[t][pos])


def transform_policy(policy: np.ndarray, t: int) -> np.ndarray:
    """Remap a flat 225-dim policy vector under transform *t*."""
    perm = TRANSFORMS[t]
    out = np.empty_like(policy)
    out[perm] = policy
    return out


def transform_board(board_cells: np.ndarray, t: int) -> np.ndarray:
    """Apply transform *t* to a (15,15) board array."""
    if t == 0:
        return board_cells.copy()
    if t == 1:
        return np.rot90(board_cells, k=3).copy()
    if t == 2:
        return np.rot90(board_cells, k=2).copy()
    if t == 3:
        return np.rot90(board_cells, k=1).copy()
    if t == 4:
        return np.fliplr(board_cells).copy()
    if t == 5:
        return np.flipud(board_cells).copy()
    if t == 6:
        return board_cells.T.copy()
    return np.fliplr(board_cells).T.copy()


def inverse_transform(t: int) -> int:
    """Return the index of the inverse transform."""
    identity = np.arange(NUM_CELLS)
    for i, tr in enumerate(TRANSFORMS):
        if np.array_equal(tr[TRANSFORMS[t]], identity):
            return i
    raise ValueError(f"No inverse found for transform {t}")
