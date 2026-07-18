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

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device, probe=None):
        self._batch_size = env_cfg.num_envs
        self.device = device
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed
        self.probe = probe

        self.venv = SyncVectorEnv([make_env(env_cfg, idx) for idx in range(self._batch_size)])

        assert isinstance(self.venv.single_observation_space, Discrete)
        self._num_categories = int(self.venv.single_observation_space.n)

        self._obs_dim = self._num_categories

        assert isinstance(self.venv.single_action_space, Discrete)
        self._action_dim = int(self.venv.single_action_space.n)

        self._prev_action = torch.zeros(
            (self._batch_size, self._action_dim),
            dtype=torch.float32,
            device=self.device,
        )

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

    # -----------------------------------------------------------------------------
    # Intuition: Why one-hot + per-environment reset are required
    # -----------------------------------------------------------------------------
    # Dreamer learns from *vectors*, not raw categorical integers. A scalar cue like
    # "2" has no geometric meaning (it's not “twice” category 1), so the RSSM and
    # encoder cannot model transitions or reward structure from integer labels.
    # Converting Discrete(N) → N-dimensional one-hot gives Dreamer a stable,
    # differentiable representation:
    #
    #       2 → [0, 0, 1, 0]
    #
    # This preserves the structure of the observation space and makes the cue
    # learnable.
    #
    # During training, Dreamer expects each environment in the vectorized batch to
    # evolve independently. SyncVectorEnv.reset() always resets *all* envs, so we
    # must manually reset only the env that terminated. This keeps transitions
    # Markovian and prevents corrupted state streams.
    #
    # Therefore:
    #   • One-hot encode categorical cues for Dreamer’s vector-based encoder.
    #   • Reset only the finished environment to preserve correct episode boundaries.
    # -----------------------------------------------------------------------------
    def _one_hot(self, cue: np.ndarray) -> torch.Tensor:
        """
        One-hot encode either:
        • a full batch of cues (shape = [batch_size])
        • a single cue (shape = [1])
        """
        if cue.shape[0] == self._batch_size:
            # Full batch
            out = torch.zeros((self._batch_size, self._num_categories), dtype=torch.float32, device=self.device)
            for i in range(self._batch_size):
                out[i, int(cue[i])] = 1.0
            return out

        elif cue.shape[0] == 1:
            # Single env reset
            out = torch.zeros((1, self._num_categories), dtype=torch.float32, device=self.device)
            out[0, int(cue[0])] = 1.0
            return out

        else:
            raise ValueError(f"Unexpected cue shape: {cue.shape}")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.venv.reset(seed=seed)
        state = self._one_hot(obs)

        self._prev_action.zero_()
        self._needs_first[:] = True

        if self.probe:
            self.probe.env_reset(obs.tolist())

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

        # PopGym RepeatPreviousEasy-v0 returns shaped rewards ±1/max_episode_steps.
        # Dreamer’s PopGym test expects the original sparse reward:
        #     +1 for correct repeat, 0 otherwise.
        # The mapping is:
        #     reward > 0  → 1
        #     reward <= 0 → 0
        # This restores the intended learning signal.
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        reward_t = torch.where(
            reward_t > 0,
            torch.ones_like(reward_t),
            torch.zeros_like(reward_t),
        ).float()

        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        # ---------------------------------------------------------------------------------------------
        # Dreamer requires fixed-length episodes -> Treat truncation as terminal
        # ---------------------------------------------------------------------------------------------
        # NOTE: Dreamer-V3 uses both `is_last` and `is_terminal` even when they appear equal.
        #
        # • `is_last`      → marks the *end of an episode* for replay slicing and RSSM state resets.
        # • `is_terminal`  → marks a *true terminal condition* for discounting, bootstrapping,
        #                    and value/return target computation.
        #
        # In fixed-length training (e.g., PopGym with max_episode_steps), both flags become True
        # on the final step. However, in variable-length environments (Atari, DMControl, robotics):
        #
        #   - `is_last` may be True due to time limits or wrapper truncation,
        #   - while `is_terminal` is only True when the environment reaches a true terminal state.
        #
        # Dreamer must keep these concepts separate to maintain correct KL dynamics, replay
        # boundaries, and value targets. Even when equal, both flags are required.

        is_terminal = terminated_t | truncated_t
        is_last = is_terminal
        is_first = self._needs_first.clone()

        actions_t = actions.detach().long().to(self.device)

        prev = torch.zeros(
            (self._batch_size, self._action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        prev[torch.arange(self._batch_size), actions_t] = 1.0

        self._prev_action = prev

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
                self._prev_action[i] = torch.zeros(self._action_dim, device=self.device)
                self._needs_first[i] = True
            else:
                self._needs_first[i] = False

        if self.probe:
            self.probe.env_step(
                obs.tolist(), reward_t.tolist(), terminated_t.tolist(), truncated_t.tolist(), is_last.tolist()
            )

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
