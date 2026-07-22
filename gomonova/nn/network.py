"""GomoNovaNet: Multi-Scale Attentive Residual Network for Gomoku.

Architecture:
  Stem (Conv 8->C) + positional encoding
  -> N x MSARBlock
  -> Policy head (local logits + global-context gate -> 225 logits)
  -> Value head (scalar in [-1, 1])
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..game.board import BOARD_SIZE, NUM_CELLS
from .blocks import MSARBlock

INPUT_CHANNELS = 8


class GomoNovaNet(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        num_blocks: int = 12,
        policy_channels: int = 64,
        value_channels: int = 32,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.channels = channels

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
        )
        self.pos_enc = nn.Parameter(torch.randn(channels, BOARD_SIZE, BOARD_SIZE) * 0.01)

        self.tower = nn.Sequential(
            *(MSARBlock(channels, se_reduction) for _ in range(num_blocks))
        )

        self.policy_conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.Conv2d(channels, policy_channels, 1, bias=False),
        )
        self.policy_local = nn.Conv2d(policy_channels, 1, 1)
        self.policy_global = nn.Sequential(
            nn.Linear(policy_channels, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, NUM_CELLS),
        )
        self.policy_gate = nn.Linear(policy_channels, 1)

        self.value_head = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.Conv2d(channels, value_channels, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(value_channels, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits [B, 225], value [B, 1])."""
        h = self.stem(x) + self.pos_enc.unsqueeze(0)
        h = self.tower(h)

        p = self.policy_conv(h)
        local_logits = self.policy_local(p).flatten(1)
        gap = p.mean(dim=(2, 3))
        global_bias = self.policy_global(gap)
        gate = torch.sigmoid(self.policy_gate(gap))
        policy_logits = local_logits + gate * global_bias

        value = self.value_head(h)
        return policy_logits, value

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
