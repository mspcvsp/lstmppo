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
import torch.nn as nn


def init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MinigridEncoder(nn.Module):
    """
    Lightweight CNN encoder for MiniGrid image observations.
    Designed for DreamerV3 warmup before CAGE #2.

    • Preserves POMDP structure (ImgObsWrapper only)
    • Uses GroupNorm for determinism
    • Uses SiLU for stable gradients
    • Uses Xavier init for stable KL dynamics
    """

    def __init__(self, latent_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        # Compute flattened size dynamically
        dummy = torch.zeros(1, 3, 7, 7)
        with torch.no_grad():
            flat_dim = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Linear(flat_dim, latent_dim)

        self.apply(init_xavier)

    def forward(self, obs):
        """
        obs: (B, H, W, C) from ImgObsWrapper
        convert to (B, C, H, W)
        """
        x = obs.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


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
