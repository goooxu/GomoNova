"""Original building blocks: Multi-Scale Attentive Residual (MSAR) block + SE attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3))
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class _ConvBranch(nn.Module):
    """BN → Mish → Conv → BN → Mish → Conv with a given kernel size."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MSARBlock(nn.Module):
    """Multi-Scale Attentive Residual block.

    Three parallel convolution branches (3x3, 5x5, 7x7) capture patterns at
    different spatial scales.  Their outputs are concatenated, fused with a
    1x1 convolution, recalibrated by SE attention, and added back as a
    residual.
    """

    def __init__(self, channels: int, se_reduction: int = 4):
        super().__init__()
        self.branch3 = _ConvBranch(channels, 3)
        self.branch5 = _ConvBranch(channels, 5)
        self.branch7 = _ConvBranch(channels, 7)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b3, b5, b7], dim=1)
        out = self.fuse(out)
        out = self.se(out)
        return F.mish(out + x)
