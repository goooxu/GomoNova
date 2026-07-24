"""Array-based MCTS tree — eliminates Python object overhead.

Replaces per-node MCTSNode objects (dict children, Python floats) with
pre-allocated numpy arrays.  Node creation drops from ~212μs to ~5μs.

Child statistics (visit_count, total_value) are indexed by child slot
(node * NUM_CELLS + child_index), NOT by node index.
"""

from __future__ import annotations

import math

import numpy as np

from ..game.board import NUM_CELLS


class FlatMCTSTree:
    """MCTS tree backed by flat numpy arrays for one root position."""

    __slots__ = (
        "max_nodes", "num_children", "child_prior",
        "child_visit", "child_value", "child_action",
        "is_expanded", "_count",
    )

    def __init__(self, max_nodes: int = 64):
        self.max_nodes = max_nodes
        total_slots = max_nodes * NUM_CELLS
        self.num_children = np.zeros(max_nodes, dtype=np.int32)
        self.is_expanded = np.zeros(max_nodes, dtype=np.bool_)
        self.child_action = np.zeros(total_slots, dtype=np.int32)
        self.child_prior = np.zeros(total_slots, dtype=np.float32)
        self.child_visit = np.zeros(total_slots, dtype=np.int32)
        self.child_value = np.zeros(total_slots, dtype=np.float64)
        self._count = 1  # node 0 = root

    def expand_node(
        self, node: int, policy: np.ndarray, legal: np.ndarray,
    ) -> None:
        start = node * NUM_CELLS
        n = len(legal)
        total = float(policy[legal].sum())
        self.child_action[start:start + n] = legal
        if total > 0:
            self.child_prior[start:start + n] = policy[legal] / total
        else:
            self.child_prior[start:start + n] = 1.0 / n
        self.num_children[node] = n
        self.is_expanded[node] = True

    def add_dirichlet(self, node: int, alpha: float, epsilon: float) -> None:
        start = node * NUM_CELLS
        n = self.num_children[node]
        if n == 0:
            return
        noise = np.random.dirichlet([alpha] * n).astype(np.float32)
        priors = self.child_prior[start:start + n]
        priors[:] = (1 - epsilon) * priors + epsilon * noise

    def best_child(self, node: int, c_puct: float) -> tuple[int, int]:
        """Returns (child_slot_index, action). Vectorized with numpy."""
        start = node * NUM_CELLS
        n = int(self.num_children[node])
        vc = self.child_visit[start:start + n].astype(np.float64)
        tv = self.child_value[start:start + n]
        prior = self.child_prior[start:start + n]

        parent_visits = max(int(vc.sum()), 1)
        sqrt_pv = math.sqrt(parent_visits)

        q = np.where(vc > 0, tv / np.maximum(vc, 1), 0.0)
        u = c_puct * prior * sqrt_pv / (1.0 + vc)
        best_i = int(np.argmax(q + u))
        return start + best_i, int(self.child_action[start + best_i])

    def backup(self, slots: list[int], value: float) -> None:
        for idx in reversed(slots):
            self.child_visit[idx] += 1
            self.child_value[idx] += value
            value = -value

    def alloc_node(self) -> int:
        idx = self._count
        self._count += 1
        return idx

    def get_visit_distribution(self, temperature: float = 1.0) -> np.ndarray:
        policy = np.zeros(NUM_CELLS, dtype=np.float32)
        n = self.num_children[0]
        for i in range(n):
            a = int(self.child_action[i])
            policy[a] = self.child_visit[i]

        if temperature < 1e-8:
            best = np.argmax(policy)
            policy[:] = 0.0
            policy[best] = 1.0
        else:
            policy = policy ** (1.0 / temperature)
            total = policy.sum()
            if total > 0:
                policy /= total
            else:
                policy[:] = 1.0 / NUM_CELLS
        return policy
