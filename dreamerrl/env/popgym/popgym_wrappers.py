"""
PopGym → Dreamer Observation Contract (SyncVectorEnv-compatible)

PopGym environments normally return dict observations:
    - obs["state"]:       environment state
    - obs["action"]:      previous action taken

BUT Gymnasium's SyncVectorEnv *collapses dict observations into arrays*.
This means obs["action"] and obs["state"] are NOT available in vectorized
mode — only a single numpy array containing the state is returned.

To support PopGym's RepeatPrevious* tasks (reward = +1 if action_t == action_{t-1}),
this wrapper tracks the previous action INTERNALLY:

    self._prev_action[i] = action taken by env i on the previous step

Dreamer receives:
    - "state":       flattened state array
    - "prev_action": previous action (tracked internally)

Dreamer’s encoder concatenates these internally, producing:
    [s0, s1, s2, s3, prev_action]

This is REQUIRED for RepeatPrevious* tasks because the reward depends on
whether the agent repeats the previous action.

AUTO‑RESET LOGIC (Dreamer-style streaming):

Dreamer trains on a continuous stream of fixed-length sequences. When an
environment finishes an episode (is_last=True):

    1. We immediately reset *only* that environment.
    2. We flatten the new initial state.
    3. We reset prev_action for that environment to 0.
    4. We stitch the reset state + prev_action into the batch output.
    5. We mark that environment as is_first=True on the NEXT step.

This produces seamless transitions across episode boundaries:

    ... → (terminal) → (reset obs) → ...

allowing Dreamer to sample fixed-length sequences without encountering
dead environments. This is the standard DreamerV3 vector-environment pattern.
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

from .popgym_preprocessing import flatten_obs


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
    Tracks prev_action internally (SyncVectorEnv does not expose it).
    """

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device):
        self._batch_size = env_cfg.num_envs
        self.device = device
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed

        self.venv = SyncVectorEnv([make_env(env_cfg, idx) for idx in range(self._batch_size)])

        # Flattened state dimension
        shape = self.venv.single_observation_space.shape
        if shape is None:
            raise RuntimeError("PopGymVecEnv: single_observation_space.shape is None")

        self._obs_dim = int(np.prod(shape))

        assert isinstance(self.venv.single_action_space, Discrete)
        self._action_dim = int(self.venv.single_action_space.n)

        # Track previous actions internally
        self._prev_action = torch.zeros(self._batch_size, dtype=torch.float32, device=self.device)

        # Track Dreamer-style "first step" markers
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

        # Flatten state
        flat_state = flatten_obs(obs, self.venv.single_observation_space)
        state = torch.as_tensor(flat_state, dtype=torch.float32, device=self.device)

        # Reset prev_action to 0
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
        # Normalize actions to shape (B,)
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions_np = actions.squeeze(-1).detach().cpu().numpy()
        else:
            actions_np = actions.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = self.venv.step(actions_np)

        # Flatten state
        flat_state = flatten_obs(obs, self.venv.single_observation_space)
        state = torch.as_tensor(flat_state, dtype=torch.float32, device=self.device)

        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        is_terminal = terminated_t
        is_last = terminated_t | truncated_t
        is_first = self._needs_first.clone()

        # Update prev_action BEFORE auto-reset
        self._prev_action = actions.detach().float().to(self.device)

        # ------------------------------------------------------------------
        # Auto-reset logic (Dreamer-style streaming)
        # ------------------------------------------------------------------
        if bool(is_last.any()):
            if self.deterministic:
                seeds = [(self.base_seed + i) if is_last[i].item() else None for i in range(self._batch_size)]
                reset_obs, _ = self.venv.reset(seed=seeds)
            else:
                reset_obs, _ = self.venv.reset()

            reset_flat_state = flatten_obs(reset_obs, self.venv.single_observation_space)
            reset_state = torch.as_tensor(reset_flat_state, dtype=torch.float32, device=self.device)

            # Replace finished envs
            state = torch.where(is_last[:, None], reset_state, state)

            # Reset prev_action for finished envs
            self._prev_action = torch.where(
                is_last,
                torch.zeros_like(self._prev_action),
                self._prev_action,
            )

            self._needs_first = is_last.clone()
        else:
            self._needs_first.zero_()

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
