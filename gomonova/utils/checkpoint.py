"""Checkpoint save/load utilities."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    iteration: int = 0,
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    state = {
        "network": network.state_dict(),
        "iteration": iteration,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    network: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    state = torch.load(path, map_location=device, weights_only=True)
    network.load_state_dict(state["network"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("iteration", 0)
