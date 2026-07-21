from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from dreamerrl.models.base_obs_encoder import BaseEncoder
from dreamerrl.models.cnn_obs_encoder import CNNObsEncoder
from dreamerrl.models.mlp_obs_encoder import MLPObsEncoder


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
# 4. Builder
# ============================================================
def build_obs_encoder(space: gym.Space, embed_dim: int = 256) -> BaseEncoder:
    # Crafter: Box(64,64,3)
    if isinstance(space, gym.spaces.Box) and len(space.shape) == 3:
        return CNNObsEncoder(space, embed_dim)

    # PopGym: Discrete → one-hot → MLP
    if isinstance(space, gym.spaces.Discrete):
        flat_dim = int(space.n)
        return MLPObsEncoder(flat_dim, embed_dim)

    # CAGE2: Dict/Tuple → flatten → MLP
    if isinstance(space, gym.spaces.Dict):
        flat_dim = sum(get_flat_obs_dim(sub) for sub in space.spaces.values())
        return MLPObsEncoder(flat_dim, embed_dim)

    if isinstance(space, gym.spaces.Tuple):
        flat_dim = sum(get_flat_obs_dim(sub) for sub in space.spaces)
        return MLPObsEncoder(flat_dim, embed_dim)

    # Generic Box (vector)
    if isinstance(space, gym.spaces.Box) and len(space.shape) == 1:
        flat_dim = int(np.prod(space.shape))
        return MLPObsEncoder(flat_dim, embed_dim)

    raise NotImplementedError(f"Unsupported observation space: {space}")
