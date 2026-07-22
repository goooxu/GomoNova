"""Input plane construction: 8 binary channels + learnable positional encoding."""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, BOARD_SIZE, Board


def board_to_planes(board: Board) -> np.ndarray:
    """Convert board state to 8-channel float32 array (8, 15, 15).

    Channels (color-relative, current player always on channel 0):
      0: current player's stones
      1: opponent's stones
      2: current player's last move
      3: current player's 2nd-to-last move
      4: opponent's last move
      5: opponent's 2nd-to-last move
      6: occupancy (all stones)
      7: turn bias (all-ones if current is Black, else zeros)
    """
    planes = np.zeros((8, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    current = board.current
    opponent = 3 - current  # BLACK=1 ↔ WHITE=2

    planes[0] = (board.cells == current).astype(np.float32)
    planes[1] = (board.cells == opponent).astype(np.float32)

    cur_last = board.last_moves_for(current, 2)
    opp_last = board.last_moves_for(opponent, 2)

    if len(cur_last) >= 1:
        r, c = divmod(cur_last[0], BOARD_SIZE)
        planes[2, r, c] = 1.0
    if len(cur_last) >= 2:
        r, c = divmod(cur_last[1], BOARD_SIZE)
        planes[3, r, c] = 1.0
    if len(opp_last) >= 1:
        r, c = divmod(opp_last[0], BOARD_SIZE)
        planes[4, r, c] = 1.0
    if len(opp_last) >= 2:
        r, c = divmod(opp_last[1], BOARD_SIZE)
        planes[5, r, c] = 1.0

    planes[6] = (board.cells != 0).astype(np.float32)

    if current == BLACK:
        planes[7] = 1.0

    return planes


def planes_to_tensor(planes: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    t = torch.from_numpy(planes).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t
