"""
FakeRolloutBuilder

Creates fully aligned synthetic rollouts for testing:

- RecurrentRolloutBuffer
- TBPTT chunking
- next_obs alignment
- aux prediction heads
- policy unrolls
- drift tests
- reward/obs/action alignment

This builder guarantees:
- obs[t] has shape (B, obs_dim)
- next_obs[t] = obs[t+1], except final timestep which is padded with zeros
- rewards, actions, masks all aligned with obs
- hidden states optionally included
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FakeRollout:
    obs: torch.Tensor  # [T, B, obs_dim]
    next_obs: torch.Tensor  # [T, B, obs_dim]
    actions: torch.Tensor  # [T, B]
    rewards: torch.Tensor  # [T, B]
    masks: torch.Tensor  # [T, B]
    h0: Optional[torch.Tensor] = None
    c0: Optional[torch.Tensor] = None


class FakeRolloutBuilder:
    def __init__(self, T: int, B: int, obs_dim: int, *, device="cpu"):
        self.T = T
        self.B = B
        self.obs_dim = obs_dim
        self.device = device

        # Defaults
        self._pattern = "range"
        self._include_hidden = False
        self._hidden_size = None

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    def with_pattern(self, pattern: str):
        """
        pattern ∈ {"range", "zeros", "random"}
        """
        self._pattern = pattern
        return self

    def with_hidden(self, hidden_size: int):
        """
        Include initial LSTM hidden states (h0, c0).
        """
        self._include_hidden = True
        self._hidden_size = hidden_size
        return self

    # ------------------------------------------------------------
    # Build rollout
    # ------------------------------------------------------------
    def build(self) -> FakeRollout:
        T, B, D = self.T, self.B, self.obs_dim
        device = self.device

        # -------------------------
        # obs
        # -------------------------
        if self._pattern == "range":
            obs = torch.arange(T * B * D, dtype=torch.float32, device=device).reshape(T, B, D)
        elif self._pattern == "zeros":
            obs = torch.zeros(T, B, D, dtype=torch.float32, device=device)
        else:  # random
            obs = torch.randn(T, B, D, device=device)

        # -------------------------
        # next_obs
        # -------------------------
        next_obs = obs.roll(-1, dims=0).clone()
        next_obs[-1] = 0.0  # final timestep padded

        # -------------------------
        # actions
        # -------------------------
        actions = torch.zeros(T, B, dtype=torch.long, device=device)

        # -------------------------
        # rewards
        # -------------------------
        rewards = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1).expand(T, B)

        # -------------------------
        # masks (all ones)
        # -------------------------
        masks = torch.ones(T, B, dtype=torch.float32, device=device)

        # -------------------------
        # optional hidden states
        # -------------------------
        if self._include_hidden:
            assert self._hidden_size is not None
            h0 = torch.zeros(B, self._hidden_size, device=device)
            c0 = torch.zeros(B, self._hidden_size, device=device)
        else:
            h0 = None
            c0 = None

        return FakeRollout(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=rewards,
            masks=masks,
            h0=h0,
            c0=c0,
        )
