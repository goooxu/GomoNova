"""Input plane construction: 16 binary channels + learnable positional encoding."""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, BOARD_SIZE, Board

NUM_HISTORY = 6
INPUT_CHANNELS = 2 + 2 * NUM_HISTORY + 2  # 16


def board_to_planes(board: Board) -> np.ndarray:
    """Convert board state to 16-channel float32 array (16, 15, 15).

    Channels (color-relative, current player always on channel 0):
      0:    current player's stones
      1:    opponent's stones
      2-7:  current player's last 6 moves (one plane each)
      8-13: opponent's last 6 moves (one plane each)
      14:   occupancy (all stones)
      15:   turn bias (all-ones if current is Black, else zeros)
    """
    planes = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    current = board.current
    opponent = 3 - current  # BLACK=1 ↔ WHITE=2

    planes[0] = (board.cells == current).astype(np.float32)
    planes[1] = (board.cells == opponent).astype(np.float32)

    cur_last = board.last_moves_for(current, NUM_HISTORY)
    opp_last = board.last_moves_for(opponent, NUM_HISTORY)

    for i, pos in enumerate(cur_last):
        r, c = divmod(pos, BOARD_SIZE)
        planes[2 + i, r, c] = 1.0
    for i, pos in enumerate(opp_last):
        r, c = divmod(pos, BOARD_SIZE)
        planes[2 + NUM_HISTORY + i, r, c] = 1.0

    planes[2 + 2 * NUM_HISTORY] = (board.cells != 0).astype(np.float32)

    if current == BLACK:
        planes[2 + 2 * NUM_HISTORY + 1] = 1.0

    return planes


def planes_to_tensor(planes: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    t = torch.from_numpy(planes).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t
