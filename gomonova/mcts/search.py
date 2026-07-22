"""MCTS search using neural network evaluation (training only).

This module is imported ONLY by training/selfplay.py.  The inference path
(gomonova/inference/) must never import from here.
"""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, Board
from ..game.rules import check_winner_at, is_legal
from ..nn.encoder import board_to_planes
from .node import MCTSNode


class MCTSSearch:
    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 400,
        c_puct: float = 2.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @torch.no_grad()
    def _evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        policy_logits, value = self.network(x)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        v = value.item()
        return policy, v

    def _get_legal_actions(self, board: Board) -> np.ndarray:
        moves = board.legal_moves()
        if board.current == BLACK:
            legal = np.array([m for m in moves if is_legal(board, int(m))], dtype=np.int64)
            if len(legal) == 0:
                return moves
            return legal
        return moves

    def search(self, board: Board, add_noise: bool = True) -> MCTSNode:
        """Run MCTS from the current board state and return the root node."""
        root = MCTSNode()
        policy, _ = self._evaluate(board)
        legal = self._get_legal_actions(board)
        root.expand(policy, legal)
        if add_noise:
            root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)

        for _ in range(self.num_simulations):
            node = root
            sim_board = board.copy()

            while node.is_expanded and node.children:
                node = node.best_child(self.c_puct)
                sim_board.play(node.action)
                winner = check_winner_at(sim_board, node.action)
                if winner is not None:
                    if sim_board.current == BLACK:
                        value = -1.0
                    else:
                        value = 1.0
                    node.backup(value)
                    break
                if sim_board.is_full():
                    node.backup(0.0)
                    break
            else:
                if not node.is_expanded:
                    policy, v = self._evaluate(sim_board)
                    legal = self._get_legal_actions(sim_board)
                    if len(legal) == 0:
                        node.backup(0.0)
                    else:
                        node.expand(policy, legal)
                        node.backup(v)

        return root

    def get_move(self, board: Board, temperature: float = 1.0) -> tuple[int, np.ndarray]:
        """Run search and select a move. Returns (move, visit_policy)."""
        root = self.search(board, add_noise=True)
        policy = root.get_visit_distribution(temperature)
        move = int(np.argmax(policy)) if temperature < 1e-8 else int(
            np.random.choice(len(policy), p=policy)
        )
        return move, policy
