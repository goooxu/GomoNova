"""Pure forward-pass inference player. NO search, NO rule intervention.

Move selection is 100% determined by the model output.  Only empty-cell
masking is applied (a physical constraint, not a rule intervention).
"""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import Board
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
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
            logits, value = self.network(x)
        policy = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        return policy

    @torch.no_grad()
    def get_move(self, board: Board) -> tuple[int, np.ndarray, float]:
        """Returns (move, full_policy, value).

        The model output directly determines the move.  Only empty
        cells are considered (physical constraint).
        """
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
            logits, value = self.network(x)
        policy = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        v = value.item()

        empty = board.legal_moves()
        masked = np.zeros(225, dtype=np.float32)
        masked[empty] = policy[empty]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            masked[empty[0]] = 1.0

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
