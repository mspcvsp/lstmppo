from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit

import popgym  # noqa: F401  # ensures PopGym registers its environments
from dreamerrl.utils.types import EnvironmentConfig

from .popgym_preprocessing import flatten_obs


def make_env(env_cfg: EnvironmentConfig, idx: int) -> Callable[[], gym.Env]:
    def thunk():
        # Reset global NumPy RNG BEFORE construction (Deck shuffle)
        if env_cfg.deterministic:
            np.random.seed(env_cfg.seed + idx)

        env = gym.make(env_cfg.env_id)
        env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)

        if env_cfg.deterministic:
            # Reseed PopGym's internal RNG if present
            rng = getattr(env.unwrapped, "rng", None)
            if rng is not None:
                setattr(env.unwrapped, "rng", np.random.default_rng(env_cfg.seed + idx))

            # Also seed Gymnasium RNG
            env.reset(seed=env_cfg.seed + idx)

        return env

    return thunk


class PopGymParallelEnv:
    """
    True parallel PopGym environments using AsyncVectorEnv.
    Each environment runs in its own process.
    """

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device):
        self.cfg = env_cfg
        self.device = device
        self.num_envs = env_cfg.num_envs
        self.env_id = env_cfg.env_id
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed

        # Build subprocess envs
        self.venv = AsyncVectorEnv([make_env(env_cfg, idx) for idx in range(self.num_envs)])

        # Expose spaces
        self.single_observation_space = self.venv.single_observation_space
        self.single_action_space = self.venv.single_action_space

        # Track "first step" markers for Dreamer-style streaming
        self._needs_first = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, seed=None):
        obs, info = self.venv.reset(seed=seed)

        obs = flatten_obs(obs, self.single_observation_space)
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        self._needs_first[:] = True

        return {
            "state": state,
            "reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "is_first": torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            "is_last": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "is_terminal": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "info": info,
        }

    def step(self, actions: torch.Tensor):
        # Normalize actions to shape (B,)
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions_np = actions.squeeze(-1).detach().cpu().numpy()
        else:
            actions_np = actions.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = self.venv.step(actions_np)

        obs = flatten_obs(obs, self.single_observation_space)
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        is_terminal = terminated_t
        is_last = terminated_t | truncated_t
        is_first = self._needs_first.clone()

        # Auto-reset ended envs
        if bool(is_last.any()):
            if self.deterministic:
                seeds = [(self.base_seed + i) if is_last[i].item() else None for i in range(self.num_envs)]
                reset_obs, _ = self.venv.reset(seed=seeds)
            else:
                reset_obs, _ = self.venv.reset()

            reset_obs = flatten_obs(reset_obs, self.single_observation_space)
            reset_state = torch.as_tensor(reset_obs, dtype=torch.float32, device=self.device)

            state = torch.where(is_last[:, None], reset_state, state)
            self._needs_first = is_last.clone()
        else:
            self._needs_first.zero_()

        return {
            "state": state,
            "reward": reward_t,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "info": info,
        }

    @property
    def batch_size(self) -> int:
        return self.num_envs

    @property
    def obs_dim(self) -> int:
        # flatten_obs produces a vector; use the observation space shape
        shape = self.single_observation_space.shape
        if shape is None:
            raise RuntimeError("single_observation_space.shape is None")
        return shape[0]

    @property
    def action_dim(self) -> int:
        space = self.single_action_space
        if not isinstance(space, Discrete):
            raise RuntimeError(f"PopGymParallelEnv only supports Discrete action spaces, got {type(space)}")
        return int(space.n)
