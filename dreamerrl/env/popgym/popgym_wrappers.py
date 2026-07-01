"""
PopGym → Dreamer Observation Contract

PopGym environments return dict observations that include:
    - obs["state"]:       the actual environment state (Box space)
    - obs["action"]:      the *previous* action taken (Discrete → scalar)

Dreamer expects a single flat observation vector, so this wrapper:
    1. Flattens only obs["state"] into [s0, s1, ..., sn]
    2. Converts obs["action"] into a 1-D tensor [prev_action]
    3. Returns both fields separately ("state", "prev_action")

The Dreamer encoder then concatenates these internally, producing:

        [s0, s1, s2, s3, prev_action]

This is REQUIRED for PopGym's RepeatPrevious* tasks, whose reward depends on whether the agent repeats the previous
action:

        reward = +1 if action_t == action_{t-1}

Without exposing prev_action, Dreamer cannot learn the rule.

Additionally, Dreamer trains on a continuous stream of transitions. When an environment finishes an episode
(is_last=True), we immediately reset *only* that environment and stitch the reset observation (state + prev_action)
into the batch. This produces seamless transitions across episode boundaries:

        ... → (terminal) → (reset obs) → ...

allowing Dreamer to sample fixed-length sequences without encountering dead environments. This is the standard
DreamerV3 vector-environment pattern.
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
    Now correctly exposes:
      - state (flattened)
      - prev_action (required for RepeatPreviousEasy)
      - reward
      - is_first / is_last / is_terminal
    """

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device):
        self._batch_size = env_cfg.num_envs
        self.device = device
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed

        self.venv = SyncVectorEnv([make_env(env_cfg, idx) for idx in range(self._batch_size)])

        # Observation dimension (flattened state only)
        self._obs_dim = int(torch.tensor(self.venv.single_observation_space.shape).prod())

        assert isinstance(self.venv.single_action_space, Discrete)
        self._action_dim = int(self.venv.single_action_space.n)

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

        # Extract previous action
        prev_action = torch.as_tensor(obs["action"], dtype=torch.float32, device=self.device)

        # Flatten only the state part
        flat_state = flatten_obs(obs["state"], self.venv.single_observation_space)
        state = torch.as_tensor(flat_state, dtype=torch.float32, device=self.device)

        self._needs_first[:] = True

        return {
            "state": state,
            "prev_action": prev_action,
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

        # Extract previous action
        prev_action = torch.as_tensor(obs["action"], dtype=torch.float32, device=self.device)

        # Flatten only the state part
        flat_state = flatten_obs(obs["state"], self.venv.single_observation_space)
        state = torch.as_tensor(flat_state, dtype=torch.float32, device=self.device)

        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        is_terminal = terminated_t
        is_last = terminated_t | truncated_t
        is_first = self._needs_first.clone()

        # ------------------------------------------------------------------
        # Auto‑reset logic (Dreamer-style streaming)
        #
        # Dreamer trains on a continuous stream of fixed-length sequences.
        # To support this, environments must NEVER stop producing transitions.
        #
        # When an env finishes an episode (is_last=True):
        #   1. We immediately call env.reset() *only for those envs*.
        #   2. We extract the new initial observation (state + prev_action).
        #   3. We stitch the reset observation into the batch output:
        #        - state[i]      ← reset_state[i]
        #        - prev_action[i]← reset_prev_action[i]
        #   4. We mark that env as `is_first=True` on the NEXT step.
        #
        # This produces a seamless transition:
        #
        #     ... → (terminal) → (reset obs) → ...
        #
        # allowing Dreamer to sample fixed-horizon sequences without ever
        # encountering a "dead" environment. All envs remain active, and
        # Dreamer sees a continuous stream of transitions across episode
        # boundaries.
        #
        # IMPORTANT:
        #   - We must replace BOTH `state` and `prev_action` for reset envs.
        #   - `is_last` marks the boundary where the reset occurs.
        #   - `is_first` marks the first transition *after* the reset.
        #
        # This is the standard DreamerV3 vector-environment pattern.
        # ------------------------------------------------------------------
        if bool(is_last.any()):
            if self.deterministic:
                seeds = [(self.base_seed + i) if is_last[i].item() else None for i in range(self._batch_size)]
                reset_obs, _ = self.venv.reset(seed=seeds)
            else:
                reset_obs, _ = self.venv.reset()

            # Extract previous action on reset
            reset_prev_action = torch.as_tensor(reset_obs["action"], dtype=torch.float32, device=self.device)

            # Flatten only the state part
            reset_flat_state = flatten_obs(reset_obs["state"], self.venv.single_observation_space)
            reset_state = torch.as_tensor(reset_flat_state, dtype=torch.float32, device=self.device)

            # Replace finished envs
            state = torch.where(is_last[:, None], reset_state, state)
            prev_action = torch.where(is_last[:, None], reset_prev_action, prev_action)

            self._needs_first = is_last.clone()
        else:
            self._needs_first.zero_()

        return {
            "state": state,
            "prev_action": prev_action,
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
