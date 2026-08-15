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
