import gymnasium as gym
import numpy as np
import torch


def get_flat_obs_dim(space: gym.Space) -> int:
    """
    Generic flattened dimension calculator for any Gymnasium observation space.
    Used by PopGym, CAGE‑2, and MLP encoders.
    MiniGrid (ImgObsWrapper) does NOT use this.
    """
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))

    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)

    if isinstance(space, gym.spaces.Dict):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces.values())

    if isinstance(space, gym.spaces.Tuple):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces)

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
