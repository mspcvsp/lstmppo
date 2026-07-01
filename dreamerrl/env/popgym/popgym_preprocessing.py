"""
PopGym Observation Preprocessing for Dreamer

PopGym environments return dict observations, but Dreamer expects a single flat tensor as input to its observation
encoder. This file provides:

    1. get_flat_obs_dim(space)
        Computes the flattened dimension of a Gymnasium observation space. Supports Box, Dict, Tuple, and Discrete
        spaces.

    2. flatten_obs(obs, space)
        Converts a structured observation (numpy) into a flat vector. Dict and Tuple spaces are flattened by
        concatenating their parts.

    3. ObsEncoder
        A Dreamer-style MLP encoder:
            - Input:  (B, flat_dim)
            - Output: (B, embed_dim)
            - SiLU activations + Xavier initialization
            - Applies symlog() before encoding

    4. build_obs_encoder(space)
        Convenience builder that computes flat_dim and constructs the encoder.

IMPORTANT FOR POPGYM:
--------------------
PopGym's RepeatPrevious* tasks require the previous action to be part of  observation. The wrapper exposes this as
"prev_action". Dreamer then receives observations shaped like:

    [s0, s1, s2, s3, prev_action]

flatten_obs() is used only for the "state" portion; "prev_action" is passed separately and concatenated internally by
Dreamer.

This preprocessing layer ensures that Dreamer receives a consistent, flattened representation of PopGym observations
suitable for RSSM encoding.
"""

import gymnasium as gym
import numpy as np
import torch


def get_flat_obs_dim(space: gym.Space) -> int:
    """
    Compute flattened observation dimension for any PopGym observation space.
    PopGym uses Box, Tuple, or Dict spaces.
    """
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))

    elif isinstance(space, gym.spaces.Dict):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces.values())

    elif isinstance(space, gym.spaces.Tuple):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces)

    else:
        raise NotImplementedError(f"Unsupported observation space: {space}")


def flatten_obs(obs, space: gym.Space) -> np.ndarray:
    """
    Flatten a PopGym observation into a (B, flat_dim) numpy array.
    PopGym vectorized envs return obs with shape (B, ...).
    """
    # Box space
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
