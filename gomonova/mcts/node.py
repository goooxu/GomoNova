"""MCTS search tree node with PUCT selection (training only)."""

from __future__ import annotations

import math

import numpy as np

from ..game.board import NUM_CELLS


class MCTSNode:
    __slots__ = (
        "parent", "action", "children", "visit_count", "total_value",
        "prior", "is_expanded",
    )

    def __init__(self, parent: MCTSNode | None = None, action: int = -1, prior: float = 0.0):
        self.parent = parent
        self.action = action
        self.children: dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        # backup() 采用「先加后翻」，子节点 q_value 是子节点轮走方（即父节点玩家的
        # 对手）的视角。父节点为自己选子，应最大化自己的价值 = -子节点 q_value。
        return -self.q_value + exploration

    def best_child(self, c_puct: float) -> MCTSNode:
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))

    def expand(self, policy: np.ndarray, legal_actions: np.ndarray) -> None:
        total = policy[legal_actions].sum()
        for action in legal_actions:
            p = policy[action] / total if total > 0 else 1.0 / len(legal_actions)
            self.children[int(action)] = MCTSNode(parent=self, action=int(action), prior=float(p))
        self.is_expanded = True

    def add_dirichlet_noise(self, alpha: float, epsilon: float) -> None:
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for i, action in enumerate(actions):
            child = self.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def backup(self, value: float) -> None:
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent

    def get_visit_distribution(self, temperature: float = 1.0) -> np.ndarray:
        policy = np.zeros(NUM_CELLS, dtype=np.float32)
        for action, child in self.children.items():
            policy[action] = child.visit_count
        if temperature == 0.0 or temperature < 1e-8:
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
