"""Pure forward-pass inference player. NO search, NO MCTS imports."""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, Board
from ..game.rules import is_legal
from ..nn.encoder import board_to_planes
from ..nn.network import GomoNovaNet


class InferencePlayer:
    def __init__(
        self,
        network: GomoNovaNet,
        device: torch.device,
        temperature: float = 0.0,
    ):
        self.network = network
        self.device = device
        self.temperature = temperature
        self.network.eval()

    @torch.no_grad()
    def get_policy(self, board: Board) -> np.ndarray:
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, value = self.network(x)
        policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return policy

    @torch.no_grad()
    def get_move(self, board: Board) -> tuple[int, np.ndarray, float]:
        """Returns (move, full_policy, value)."""
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, value = self.network(x)
        policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        v = value.item()

        moves = board.legal_moves()
        if board.current == BLACK:
            legal = np.array([m for m in moves if is_legal(board, int(m))])
            if len(legal) == 0:
                legal = moves
        else:
            legal = moves

        masked = np.zeros(225, dtype=np.float32)
        masked[legal] = policy[legal]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            masked[legal[0]] = 1.0

        if self.temperature < 1e-8:
            move = int(np.argmax(masked))
        else:
            tempered = masked ** (1.0 / self.temperature)
            tempered /= tempered.sum()
            move = int(np.random.choice(225, p=tempered))

        return move, masked, v

    def top_k(self, board: Board, k: int = 3) -> list[tuple[int, float]]:
        _, policy, _ = self.get_move(board)
        top_indices = np.argsort(policy)[::-1][:k]
        return [(int(idx), float(policy[idx])) for idx in top_indices if policy[idx] > 0]
