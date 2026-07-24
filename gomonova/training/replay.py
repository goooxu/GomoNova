"""Ring-buffer replay with pre-allocated arrays.

Stores (planes, mcts_policy, move, outcome) per sample.
mcts_policy is all-zeros for pure-policy (non-MCTS) positions.
"""

from __future__ import annotations

import numpy as np

from ..game.board import NUM_CELLS
from ..nn.encoder import INPUT_CHANNELS


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.planes = np.zeros(
            (capacity, INPUT_CHANNELS, 15, 15), dtype=np.float32
        )
        self.mcts_policy = np.zeros((capacity, NUM_CELLS), dtype=np.float32)
        self.moves = np.zeros(capacity, dtype=np.int64)
        self.outcomes = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._idx = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        planes: np.ndarray,
        move: int,
        outcome: float,
        mcts_policy: np.ndarray | None = None,
    ) -> None:
        i = self._idx
        self.planes[i] = planes
        self.moves[i] = move
        self.outcomes[i] = outcome
        if mcts_policy is not None:
            self.mcts_policy[i] = mcts_policy
        else:
            self.mcts_policy[i] = 0.0
        self._idx = (i + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def add_batch(
        self,
        samples: list[tuple[np.ndarray, int, float, np.ndarray | None]],
    ) -> None:
        for planes, move, outcome, mcts_pol in samples:
            self.add(planes, move, outcome, mcts_pol)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch. Returns (planes, moves, outcomes, mcts_policy)."""
        if self._size == 0:
            raise RuntimeError("Replay buffer is empty")

        indices = np.random.choice(
            self._size, size=min(batch_size, self._size), replace=False
        )
        return (
            self.planes[indices],
            self.moves[indices],
            self.outcomes[indices],
            self.mcts_policy[indices],
        )

    def clear(self) -> None:
        self._size = 0
        self._idx = 0
