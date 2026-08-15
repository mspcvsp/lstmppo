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

from typing import Any, Dict, Optional

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit
from minigrid.wrappers import ImgObsWrapper

from dreamerrl.env.env import EnvInterface
from dreamerrl.utils.types import EnvironmentConfig


def make_env(env_cfg, idx):
    """
    Minigrid environment construction (design intent for CAGE‑2 warmup)

    We use MiniGrid as a lightweight POMDP to warm up DreamerV3 before training on TTCP CAGE‑2. The design choices here
    are intentional:

    1. Preserve partial observability (POMDP)
         - MiniGrid’s default agent view is a 7×7 egocentric window.
         - This matches CAGE‑2’s partial observability and forces the RSSM to perform latent-state inference.
         - Therefore we DO NOT use FullyObsWrapper (it would break the POMDP).

    2. Remove mission text
         - The default observation Dict includes a "mission" string backed by MissionSpace, which is not a Gymnasium
         space.
         - This breaks AsyncVectorEnv shared-memory and cannot be flattened.
         - Mission text is irrelevant for CAGE‑2, so we strip it entirely.

    3. Convert symbolic grid → RGB image
         - ImgObsWrapper converts the agent’s local symbolic grid into a Box(H,W,3) image, which is easy to encode with
         Dreamer’s CNN encoder.
         - This keeps the POMDP intact while removing Dict/MissionSpace.

    4. Use SyncVectorEnv instead of AsyncVectorEnv
         - MiniGrid’s Dict observation (with MissionSpace) is incompatible with Gymnasium’s shared-memory
         multiprocessing.
        - SyncVectorEnv avoids shared memory and works reliably for MiniGrid.
        - MiniGrid is lightweight, so SyncVectorEnv is fast enough.

    Summary:
        - POMDP preserved
        - Mission removed
        - Pure image observations
        - Dreamer-friendly encoder input
        - Deterministic, vectorized, stable

    This wrapper produces a clean, Dreamer-compatible MiniGrid environment that structurally resembles CAGE‑2 (partial
    observability, discrete actions, event-driven transitions) without the complexity of cyber autonomy.
    """

    def thunk():
        env = gym.make(env_cfg.env_id)

        """
        MissionSpace breaks Dreamer even with an image encoder. ImgObsWrapper removes MissionSpace while preserving the
        POMDP.
        """
        env = ImgObsWrapper(env)
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
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

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
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

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
                state[i] = torch.as_tensor(obs_i[0], dtype=torch.float32, device=self.device)
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
