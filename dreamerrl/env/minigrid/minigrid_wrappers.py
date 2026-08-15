"""
Minigrid Dreamer-Native Vector Env Wrapper

Design Rationale
----------------

Minigrid observations are images (e.g., 7×7×3) that already carry geometric
structure. Unlike PopGym’s categorical cues, these do NOT require one-hot
encoding or prev_action tracking in the wrapper.

Dreamer’s encoder can be:

    • A shallow CNN that consumes (B, H, W, C) tensors, or
    • An MLP that consumes flattened vectors (B, H*W*C).

This wrapper:

    • Uses SyncVectorEnv for fast batched Minigrid.
    • Flattens observations into vectors for Dreamer.
    • Handles per-environment resets to preserve correct episode boundaries.
"""

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit

from dreamerrl.env.env import EnvInterface
from dreamerrl.utils.types import EnvironmentConfig

from .minigrid_preprocessing import flatten_obs


def make_env(env_cfg: EnvironmentConfig, idx: int) -> Callable[[], gym.Env]:
    def thunk():
        if env_cfg.deterministic:
            np.random.seed(env_cfg.seed + idx)

        env = gym.make(env_cfg.env_id)
        env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)

        if env_cfg.deterministic:
            env.reset(seed=env_cfg.seed + idx)

        return env

    return thunk


class MinigridVecEnv(EnvInterface):
    """
    Dreamer-native vector env wrapper for Minigrid.

    Observations are flattened images; actions are discrete indices.
    """

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device, probe=None):
        self._batch_size = env_cfg.num_envs
        self.device = device
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed
        self.probe = probe

        self.venv = SyncVectorEnv([make_env(env_cfg, idx) for idx in range(self._batch_size)])

        self._obs_space = self.venv.single_observation_space
        self._action_space = self.venv.single_action_space

        if not isinstance(self._action_space, Discrete):
            raise RuntimeError(f"MinigridVecEnv only supports Discrete action spaces, got {type(self._action_space)}")

        if self._obs_space.shape is None:
            raise RuntimeError("MiniGrid observation space has no shape")

        self._obs_dim = int(np.prod(self._obs_space.shape))
        self._action_dim = int(self._action_space.n)

        self._needs_first = torch.ones(self._batch_size, dtype=torch.bool, device=self.device)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.venv.reset(seed=seed)

        obs_flat = flatten_obs(obs, self._obs_space)
        state = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)

        self._needs_first[:] = True

        if self.probe:
            self.probe.env_reset(obs.tolist())

        return {
            "state": state,
            "reward": torch.zeros(self._batch_size, dtype=torch.float32, device=self.device),
            "is_first": torch.ones(self._batch_size, dtype=torch.bool, device=self.device),
            "is_last": torch.zeros(self._batch_size, dtype=torch.bool, device=self.device),
            "is_terminal": torch.zeros(self._batch_size, dtype=torch.bool, device=self.device),
            "info": info,
        }

    def step(self, actions: torch.Tensor) -> Dict[str, Any]:
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions_np = actions.squeeze(-1).detach().cpu().numpy()
        else:
            actions_np = actions.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = self.venv.step(actions_np)

        obs_flat = flatten_obs(obs, self._obs_space)
        state = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)

        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        # Separate episode end vs true terminal if needed
        is_terminal = terminated_t | truncated_t
        is_last = is_terminal
        is_first = self._needs_first.clone()

        # Per-environment reset
        for i in range(self._batch_size):
            if is_last[i]:
                if self.deterministic:
                    obs_i, _ = self.venv.envs[i].reset(seed=self.base_seed + i)
                else:
                    obs_i, _ = self.venv.envs[i].reset()

                obs_i = np.expand_dims(obs_i, axis=0)  # (1, H, W, C)
                state_i = flatten_obs(obs_i, self._obs_space)
                state[i] = torch.as_tensor(state_i[0], dtype=torch.float32, device=self.device)
                self._needs_first[i] = True
            else:
                self._needs_first[i] = False

        if self.probe:
            self.probe.env_step(
                obs.tolist(), reward_t.tolist(), terminated_t.tolist(), truncated_t.tolist(), is_last.tolist()
            )

        return {
            "state": state,
            "reward": reward_t,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "info": info,
        }

    def action_mask(self):
        return None

    def get_episode_stats(self):
        return {}
