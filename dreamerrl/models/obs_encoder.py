from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from dreamerrl.utils.transforms import symlog


# ============================================================
# 1. Compute flat observation dimension
# ============================================================
def get_flat_obs_dim(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, gym.spaces.Dict):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces.values())
    elif isinstance(space, gym.spaces.Tuple):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces)
    elif isinstance(space, gym.spaces.Discrete):
        assert space.n is not None, "Discrete space must have a defined 'n'"
        return int(space.n)
    else:
        raise NotImplementedError(f"Unsupported observation space: {space!r}")


# ============================================================
# 2. Flatten obs (numpy, for env-side use if needed)
# ============================================================
def flatten_obs(obs: Any, space: gym.Space) -> np.ndarray:
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs, dtype=np.float32)
        return obs.reshape(obs.shape[0], -1)
    elif isinstance(space, gym.spaces.Dict):
        parts = [flatten_obs(obs[k], sub) for k, sub in space.spaces.items()]
        return np.concatenate(parts, axis=-1)
    elif isinstance(space, gym.spaces.Tuple):
        parts = [flatten_obs(obs[i], sub) for i, sub in enumerate(space.spaces)]
        return np.concatenate(parts, axis=-1)
    elif isinstance(space, gym.spaces.Discrete):
        return np.asarray(obs, dtype=np.float32).reshape(-1, 1)
    else:
        raise NotImplementedError(f"Unsupported observation type: {type(obs)}")


# ============================================================
# 3. Dreamer ObsEncoder (MLP with SiLU + Xavier)
# ============================================================
class ObsEncoder(nn.Module):
    """
    Dreamer-style observation encoder.

    - Input:  (B, flat_dim) tensor
    - Output: (B, embed_dim) tensor
    - Uses SiLU activations and Xavier initialization.
    """

    def __init__(self, flat_dim: int, embed_dim: int = 256) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

        self.output_size: int = embed_dim
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2:
            raise ValueError(...)
        obs = symlog(obs)
        return self.net(obs)


# ============================================================
# 4. Builder
# ============================================================
def build_obs_encoder(space: gym.Space, embed_dim: int = 256) -> ObsEncoder:
    flat_dim = get_flat_obs_dim(space)
    return ObsEncoder(flat_dim, embed_dim)
