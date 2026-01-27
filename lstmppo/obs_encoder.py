# obs_encoder.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 1. Compute flat observation dimension from any Gym space
# ============================================================


def get_flat_obs_dim(space: gym.Space) -> int:
    """
    Recursively compute the flattened observation dimension for:
    - Box
    - Dict
    - Tuple
    - Nested structures
    """
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))

    elif isinstance(space, gym.spaces.Dict):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces.values())

    elif isinstance(space, gym.spaces.Tuple):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces)

    else:
        raise NotImplementedError(f"Unsupported observation space: {space}")


# ============================================================
# 2. Flatten raw observations into a numpy array (env wrapper)
# ============================================================


def flatten_obs(obs, space: gym.Space) -> np.ndarray:
    """
    Convert an observation (possibly dict/tuple/nested) into a flat
    numpy array of shape (N, flat_obs_dim).

    This is used inside RecurrentVecEnvWrapper BEFORE converting to torch.
    """
    if isinstance(space, gym.spaces.Box):
        # obs: (N, *shape)
        return obs.reshape(obs.shape[0], -1)

    elif isinstance(space, gym.spaces.Dict):
        # obs: dict of arrays
        parts = []
        for key, subspace in space.spaces.items():
            sub = flatten_obs(obs[key], subspace)
            parts.append(sub)
        return np.concatenate(parts, axis=-1)

    elif isinstance(space, gym.spaces.Tuple):
        # obs: tuple of arrays
        parts = []
        for i, subspace in enumerate(space.spaces):
            sub = flatten_obs(obs[i], subspace)
            parts.append(sub)
        return np.concatenate(parts, axis=-1)

    else:
        raise NotImplementedError(f"Unsupported observation type: {type(obs)}")


# ============================================================
# 3. PyTorch encoder module for the policy
# ============================================================


class FlatObsEncoder(nn.Module):
    """
    Simple identity encoder: input is already flat.
    """

    def __init__(self, flat_dim: int):
        super().__init__()
        self.output_size = flat_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, flat_dim) or (B, T, flat_dim)
        return obs


def build_obs_encoder(space: gym.Space | None, flat_dim: int) -> nn.Module:
    """
    Build a PyTorch encoder for the policy.
    Since the env wrapper already flattens observations, this is trivial.
    """
    if space is None or flat_dim == 0:
        return nn.Identity()
    else:
        return FlatObsEncoder(flat_dim)
