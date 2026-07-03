"""
Design Rationale
----------------

1. Dreamer requires *vector* observations, not raw categorical integers.

   PopGym’s RepeatPrevious* environments emit a categorical cue (e.g., 0–3).
   SyncVectorEnv collapses dict observations into a single integer, which has
   no geometric meaning to Dreamer’s encoder or RSSM. Dreamer learns from
   continuous vectors, not discrete labels, so the cue must be converted into
   a one-hot vector:

       2 → [0, 0, 1, 0]

   This preserves the structure of the observation space and gives Dreamer a
   stable, differentiable representation that the encoder and RSSM can model.

   Therefore:
       Discrete(N) → N-dimensional one-hot vector
       obs_dim = space.n

2. Dreamer requires *per-environment* resets, not batch resets.

   SyncVectorEnv.reset() always resets the entire batch, even if only one
   environment terminated. Mixing full-batch reset observations with selective
   replacement (e.g., torch.where) produces inconsistent transitions and
   non-Markovian behavior. Dreamer’s sequence stitching assumes each
   environment evolves independently, so only the environments that actually
   terminated must be reset.

   Therefore:
       - Detect which envs finished (is_last[i] == True)
       - Call envs[i].reset() directly
       - Update only state[i] and prev_action[i]
       - Leave all other envs untouched

   This preserves correct episode boundaries and produces stable transition
   streams for Dreamer’s world model.

Together, these two design choices ensure:
   • Dreamer receives meaningful, learnable observations
   • Episode boundaries are handled correctly
   • prev_action is aligned with the reward rule
   • RepeatPrevious* tasks become solvable and stable
"""

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit

import popgym  # noqa: F401
from dreamerrl.env.env import EnvInterface
from dreamerrl.utils.types import EnvironmentConfig


def make_env(env_cfg: EnvironmentConfig, idx: int) -> Callable[[], gym.Env]:
    def thunk():
        if env_cfg.deterministic:
            np.random.seed(env_cfg.seed + idx)

        env = gym.make(env_cfg.env_id)
        env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)

        if env_cfg.deterministic:
            rng = getattr(env.unwrapped, "rng", None)
            if rng is not None:
                setattr(env.unwrapped, "rng", np.random.default_rng(env_cfg.seed + idx))
            env.reset(seed=env_cfg.seed + idx)

        return env

    return thunk


class PopGymVecEnv(EnvInterface):
    """
    Dreamer-native vector env wrapper for PopGym.
    Tracks prev_action internally and reconstructs one-hot state cues.
    """

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device):
        self._batch_size = env_cfg.num_envs
        self.device = device
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed

        self.venv = SyncVectorEnv([make_env(env_cfg, idx) for idx in range(self._batch_size)])

        assert isinstance(self.venv.single_observation_space, Discrete)
        self._num_categories = int(self.venv.single_observation_space.n)

        self._obs_dim = self._num_categories

        assert isinstance(self.venv.single_action_space, Discrete)
        self._action_dim = int(self.venv.single_action_space.n)

        self._prev_action = torch.zeros(self._batch_size, dtype=torch.float32, device=self.device)
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

    def _one_hot(self, cue: np.ndarray) -> torch.Tensor:
        out = torch.zeros((self._batch_size, self._num_categories), dtype=torch.float32, device=self.device)
        for i in range(self._batch_size):
            out[i, int(cue[i])] = 1.0
        return out

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.venv.reset(seed=seed)
        state = self._one_hot(obs)

        self._prev_action.zero_()
        self._needs_first[:] = True

        return {
            "state": state,
            "prev_action": self._prev_action.clone(),
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

        state = self._one_hot(obs)

        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        is_terminal = terminated_t
        is_last = terminated_t | truncated_t
        is_first = self._needs_first.clone()

        self._prev_action = actions.detach().float().to(self.device)

        # ------------------------------------------------------------------
        # Correct per-environment reset
        # ------------------------------------------------------------------
        for i in range(self._batch_size):
            if is_last[i]:
                if self.deterministic:
                    obs_i, _ = self.venv.envs[i].reset(seed=self.base_seed + i)
                else:
                    obs_i, _ = self.venv.envs[i].reset()

                state[i] = self._one_hot(np.array([obs_i]))[0]
                self._prev_action[i] = 0.0
                self._needs_first[i] = True
            else:
                self._needs_first[i] = False

        return {
            "state": state,
            "prev_action": self._prev_action.clone(),
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
