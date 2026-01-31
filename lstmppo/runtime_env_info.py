# lstmppo/runtime_env_info.py

from dataclasses import dataclass

import gymnasium as gym
from gymnasium.spaces import Discrete

from .obs_encoder import get_flat_obs_dim


@dataclass
class RuntimeEnvInfo:
    obs_space: gym.Space | None
    flat_obs_dim: int
    action_dim: int

    @classmethod
    def from_env(cls, env: gym.Env):
        obs_space = env.observation_space
        flat_obs_dim = get_flat_obs_dim(obs_space)

        if isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
            assert isinstance(action_dim, int)

        else:
            raise ValueError("Only discrete action spaces supported")

        return cls(
            obs_space=obs_space,
            flat_obs_dim=flat_obs_dim,
            action_dim=action_dim,
        )
