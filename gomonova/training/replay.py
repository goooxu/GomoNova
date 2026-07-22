"""Ring-buffer replay for (planes, move, outcome) samples."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.planes: list[np.ndarray] = []
        self.moves: list[int] = []
        self.outcomes: list[float] = []
        self._idx = 0

    def __len__(self) -> int:
        return len(self.planes)

    def add(self, planes: np.ndarray, move: int, outcome: float) -> None:
        if len(self.planes) < self.capacity:
            self.planes.append(planes)
            self.moves.append(move)
            self.outcomes.append(outcome)
        else:
            self.planes[self._idx] = planes
            self.moves[self._idx] = move
            self.outcomes[self._idx] = outcome
        self._idx = (self._idx + 1) % self.capacity

    def add_batch(self, samples: list[tuple[np.ndarray, int, float]]) -> None:
        for planes, move, outcome in samples:
            self.add(planes, move, outcome)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch. Returns (planes, moves, outcomes)."""
        n = len(self.planes)
        if n == 0:
            raise RuntimeError("Replay buffer is empty")

        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        planes_batch = np.stack([self.planes[i] for i in indices])
        moves_batch = np.array([self.moves[i] for i in indices], dtype=np.int64)
        outcomes_batch = np.array([self.outcomes[i] for i in indices], dtype=np.float32)
        return planes_batch, moves_batch, outcomes_batch

    def clear(self) -> None:
        self.planes.clear()
        self.moves.clear()
        self.outcomes.clear()
        self._idx = 0
