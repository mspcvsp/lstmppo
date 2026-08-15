from __future__ import annotations

import gymnasium as gym
import numpy as np

from dreamerrl.models.base_obs_encoder import BaseEncoder
from dreamerrl.models.cnn_obs_encoder import CNNObsEncoder
from dreamerrl.models.mlp_obs_encoder import MLPObsEncoder
from dreamerrl.env.obs_utils import get_flat_obs_dim


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
