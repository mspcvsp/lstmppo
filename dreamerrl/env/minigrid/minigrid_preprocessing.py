"""
Minigrid Observation Preprocessing for Dreamer

Minigrid environments return image-like observations (typically Box spaces with
shape (H, W, C)). Dreamer can either:

    • Flatten these into vectors for an MLP encoder, or
    • Consume them as images via a CNN encoder defined elsewhere.

This file provides:

    1. get_flat_obs_dim(space)
        Computes the flattened dimension of a Gymnasium observation space.
        Supports Box, Dict, Tuple, and Discrete spaces.

    2. flatten_obs(obs, space)
        Converts a structured observation (numpy) into a flat vector. Dict and
        Tuple spaces are flattened by concatenating their parts.

    3. to_tensor(obs, device)
        Converts flattened numpy observations into torch tensors.

Unlike PopGym, Minigrid does NOT require one-hot encoding of categorical cues
or prev_action tracking in the wrapper. Observations are already geometric
(images), and actions are handled as integer indices.
"""

import gymnasium as gym
import numpy as np
import torch


def get_flat_obs_dim(space: gym.Space) -> int:
    """
    Compute flattened observation dimension for any Minigrid observation space.
    Minigrid typically uses Box (image), but Dict/Tuple are supported.
    """
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))

    elif isinstance(space, gym.spaces.Dict):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces.values())

    elif isinstance(space, gym.spaces.Tuple):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces)

    elif isinstance(space, gym.spaces.Discrete):
        # Rare for Minigrid, but supported
        return 1

    else:
        raise NotImplementedError(f"Unsupported observation space: {space}")


def flatten_obs(obs, space: gym.Space) -> np.ndarray:
    """
    Flatten a Minigrid observation into a (B, flat_dim) numpy array.

    For image observations (Box), this reshapes (B, H, W, C) → (B, H*W*C).
    Dict/Tuple spaces are concatenated along the last dimension.
    """
    # Box space (images or vectors)
    if isinstance(space, gym.spaces.Box):
        return obs.reshape(obs.shape[0], -1)

    # Dict space
    elif isinstance(space, gym.spaces.Dict):
        parts = []
        for key, subspace in space.spaces.items():
            parts.append(flatten_obs(obs[key], subspace))
        return np.concatenate(parts, axis=-1)

    # Tuple space
    elif isinstance(space, gym.spaces.Tuple):
        parts = []
        for i, subspace in enumerate(space.spaces):
            parts.append(flatten_obs(obs[i], subspace))
        return np.concatenate(parts, axis=-1)

    # Discrete space: vector env gives shape (B,), make it (B, 1)
    elif isinstance(space, gym.spaces.Discrete):
        obs_arr = np.asarray(obs, dtype=np.float32)
        return obs_arr.reshape(obs_arr.shape[0], 1)

    # FINAL FALLBACK: already a flat numpy array
    elif isinstance(obs, np.ndarray):
        return obs.reshape(obs.shape[0], -1)

    else:
        raise NotImplementedError(f"Unsupported observation type: {type(obs)}")


def to_tensor(obs, device):
    """
    Convert flattened numpy obs → torch tensor.
    """
    return torch.tensor(obs, dtype=torch.float32, device=device)
