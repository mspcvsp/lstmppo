from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AuxHeadConfig:
    """
    Shared config for auxiliary heads.

    in_dim:  dimension of concatenated (h_t, z_t)
    hidden:  hidden layer size
    out_dim: output dimension (1 for scalar, N for vector)
    """

    in_dim: int
    hidden: int
    out_dim: int


class BaseAuxHead(nn.Module):
    def __init__(self, cfg: AuxHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.out_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        h: (B * L, deter_size)
        z: (B * L, z_dim) or (B * L, num_classes, stoch_size) flattened
        """
        if z.dim() > 2:
            z = z.view(z.size(0), -1)

        x = torch.cat([h, z], dim=-1)  # (B * L, in_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def loss_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Default: scalar regression with MSE.
        Override for classification / multi‑dim outputs.
        """
        # logits: (B * L, out_dim) or (B, L, out_dim)
        if logits.dim() == 3:
            B, L, D = logits.shape
            logits = logits.view(B * L, D)
            target = target.view(B * L, D)

        return F.mse_loss(logits, target)


# ---------------------------------------------------------------------
# Novelty head: predicts scalar novelty per step
# ---------------------------------------------------------------------
class NoveltyHead(BaseAuxHead):
    """
    Predicts scalar novelty per time step:
        novelty[t] ≈ |obs[t+1] - obs[t]| or similar proxy.
    """

    # Uses BaseAuxHead MSE loss (scalar regression).


# ---------------------------------------------------------------------
# Reachability head: predicts scalar reachability (0/1)
# ---------------------------------------------------------------------
class ReachabilityHead(BaseAuxHead):
    """
    Predicts reachability / continuation:
        reachability[t] ≈ 1 - is_terminal[t]
    """

    def loss_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Binary classification with BCE.
        if logits.dim() == 3:
            B, L, D = logits.shape
            logits = logits.view(B * L, D)
            target = target.view(B * L, D)

        return F.binary_cross_entropy_with_logits(logits, target)


# ---------------------------------------------------------------------
# Affordance head: predicts action affordances (vector over actions)
# ---------------------------------------------------------------------
class AffordanceHead(BaseAuxHead):
    """
    Predicts which actions are possible / useful:
        affordance[t] ∈ R^{action_dim}
    """

    def loss_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Multi‑label affordance: BCE per action dimension.
        if logits.dim() == 3:
            B, L, D = logits.shape
            logits = logits.view(B * L, D)
            target = target.view(B * L, D)

        return F.binary_cross_entropy_with_logits(logits, target)


# ---------------------------------------------------------------------
# Skill head: predicts skill / option activation (vector)
# ---------------------------------------------------------------------
class SkillHead(BaseAuxHead):
    """
    Predicts which latent skill/option is active:
        skill[t] ∈ R^{num_skills}
    """

    def loss_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Multi‑label or soft assignment over skills.
        if logits.dim() == 3:
            B, L, D = logits.shape
            logits = logits.view(B * L, D)
            target = target.view(B * L, D)

        return F.mse_loss(logits, target)


# ---------------------------------------------------------------------
# Resource head: predicts scalar resource change (reward‑like)
# ---------------------------------------------------------------------
class ResourceHead(BaseAuxHead):
    """
    Predicts resource dynamics:
        resource[t] ≈ Δresources or reward[t]
    """

    # Uses BaseAuxHead MSE loss (scalar regression).


# ---------------------------------------------------------------------
# Factory to build heads given latent + network config
# ---------------------------------------------------------------------
def make_aux_heads(
    deter_size: int,
    z_dim: int,
    action_dim: int,
    num_skills: int,
    hidden: int = 256,
):
    """
    Returns a dict of auxiliary heads:
        novelty, reachability, affordance, skill, resource
    """
    in_dim = deter_size + z_dim

    heads = {
        "novelty": NoveltyHead(AuxHeadConfig(in_dim=in_dim, hidden=hidden, out_dim=1)),
        "reachability": ReachabilityHead(AuxHeadConfig(in_dim=in_dim, hidden=hidden, out_dim=1)),
        "affordance": AffordanceHead(AuxHeadConfig(in_dim=in_dim, hidden=hidden, out_dim=action_dim)),
        "skill": SkillHead(AuxHeadConfig(in_dim=in_dim, hidden=hidden, out_dim=num_skills)),
        "resource": ResourceHead(AuxHeadConfig(in_dim=in_dim, hidden=hidden, out_dim=1)),
    }

    return heads
