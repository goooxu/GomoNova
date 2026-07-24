"""Neural network trainer: outcome-weighted CE + value MSE, BF16 mixed precision."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..nn.losses import total_loss
from .replay import ReplayBuffer


class Trainer:
    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        lr: float = 2e-3,
        lr_min: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_iters: int = 5,
        total_iters: int = 400,
        grad_clip: float = 1.0,
        use_amp: bool = True,
    ):
        self.network = network
        self.device = device
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device.type == "cuda"

        self.optimizer = torch.optim.AdamW(
            network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def get_lr(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            return self.lr * (iteration + 1) / self.warmup_iters
        progress = (iteration - self.warmup_iters) / max(self.total_iters - self.warmup_iters, 1)
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(progress * math.pi))

    def update_lr(self, iteration: int) -> float:
        lr = self.get_lr(iteration)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def train_step(self, replay: ReplayBuffer, batch_size: int) -> tuple[float, float, float]:
        planes, moves, outcomes, mcts_pol = replay.sample(batch_size)
        net = self.network.module if hasattr(self.network, "module") else self.network
        dtype = next(net.parameters()).dtype
        x = torch.from_numpy(planes).to(device=self.device, dtype=dtype)
        move_indices = torch.from_numpy(moves).to(self.device)
        outcome_values = torch.from_numpy(outcomes).to(self.device)
        mcts_tensor = torch.from_numpy(mcts_pol).to(self.device)

        self.optimizer.zero_grad()
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            logits, value_pred = self.network(x)
            loss, p_loss, v_loss = total_loss(
                logits, value_pred, move_indices, outcome_values,
                self.network, mcts_policy=mcts_tensor,
            )

        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item(), p_loss.item(), v_loss.item()

    def train_epoch(self, replay: ReplayBuffer, batch_size: int, steps: int) -> dict:
        total_loss_sum = 0.0
        p_loss_sum = 0.0
        v_loss_sum = 0.0
        for _ in range(steps):
            tl, pl, vl = self.train_step(replay, batch_size)
            total_loss_sum += tl
            p_loss_sum += pl
            v_loss_sum += vl
        return {
            "loss": total_loss_sum / steps,
            "policy_loss": p_loss_sum / steps,
            "value_loss": v_loss_sum / steps,
        }
