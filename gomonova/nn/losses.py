"""Training losses: MCTS KL divergence + outcome-weighted CE + value MSE.

MCTS-guided moves use KL divergence against the visit distribution.
Pure-policy moves use outcome-weighted cross-entropy (self-imitation).
Entropy bonus prevents policy collapse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_kl_divergence(
    logits: torch.Tensor,
    mcts_policy: torch.Tensor,
) -> torch.Tensor:
    """KL(mcts_policy ∥ network_policy) for MCTS-guided positions."""
    log_probs = F.log_softmax(logits, dim=1)
    target = mcts_policy.clamp(min=1e-8)
    return (target * (target.log() - log_probs)).sum(dim=1).mean()


def policy_weighted_ce(
    logits: torch.Tensor,
    moves: torch.Tensor,
    outcomes: torch.Tensor,
    entropy_weight: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outcome-weighted cross-entropy on played moves.

    Weight = 1.0 for wins, 0.1 for losses, 0.3 for draws.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    selected_log_probs = log_probs.gather(1, moves.unsqueeze(1)).squeeze(1)

    weights = torch.where(outcomes > 0.5, torch.ones_like(outcomes),
              torch.where(outcomes < -0.5, torch.full_like(outcomes, 0.1),
                          torch.full_like(outcomes, 0.3)))

    ce_loss = -(selected_log_probs * weights).mean()
    entropy = -(probs * log_probs).sum(dim=1).mean()
    policy_loss = ce_loss - entropy_weight * entropy

    return policy_loss, entropy


def value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.squeeze(-1), target)


def total_loss(
    logits: torch.Tensor,
    value_pred: torch.Tensor,
    moves: torch.Tensor,
    outcomes: torch.Tensor,
    model: torch.nn.Module,
    mcts_policy: torch.Tensor | None = None,
    l2_weight: float = 1e-4,
    entropy_weight: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss supporting both MCTS and pure-policy samples.

    When mcts_policy is provided (B, 225), samples with non-zero MCTS
    policy use KL divergence; the rest use outcome-weighted CE.
    """
    if mcts_policy is not None:
        has_mcts = mcts_policy.sum(dim=1) > 0.5
        if has_mcts.any():
            kl = policy_kl_divergence(logits[has_mcts], mcts_policy[has_mcts])
        else:
            kl = torch.tensor(0.0, device=logits.device)
        if (~has_mcts).any():
            ce, entropy = policy_weighted_ce(
                logits[~has_mcts], moves[~has_mcts],
                outcomes[~has_mcts], entropy_weight,
            )
        else:
            ce = torch.tensor(0.0, device=logits.device)
        p_loss = kl + ce
    else:
        p_loss, entropy = policy_weighted_ce(logits, moves, outcomes, entropy_weight)

    v_loss = value_loss(value_pred, outcomes)
    l2 = sum((p ** 2).sum() for p in model.parameters())
    loss = p_loss + v_loss + l2_weight * l2
    return loss, p_loss, v_loss
