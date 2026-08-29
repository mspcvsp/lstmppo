from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class EnvInterface(ABC):
    """
    Unified interface for vectorized environments.

    Dreamer-native contract:
      - state-based observations
      - explicit episode boundary flags
    """

    # ---------------------------------------------------------
    # Core API
    # ---------------------------------------------------------

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Reset environment(s).

        Returns:
        {
            "state": Tensor (B, obs_dim)
            "reward": Tensor (B,)              # zeros
            "is_first": Tensor (B,)            # True
            "is_last": Tensor (B,)             # False
            "is_terminal": Tensor (B,)         # False
            "info": optional dict
        }
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Step environment(s).

        Args:
            actions: Tensor (B,) of discrete actions

        Returns:
        {
            "state": Tensor (B, obs_dim)
            "reward": Tensor (B,)
            "is_first": Tensor (B,)
            "is_last": Tensor (B,)
            "is_terminal": Tensor (B,)
            "info": optional dict
        }
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    @abstractmethod
    def obs_space(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def batch_size(self) -> int:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Optional: Action masking
    # ---------------------------------------------------------

    def action_mask(self) -> Optional[torch.Tensor]:
        return None

    # ---------------------------------------------------------
    # Optional: Episode stats
    # ---------------------------------------------------------

    def get_episode_stats(self) -> Dict[str, Any]:
        return {}


class DreamerVecWrapper(EnvInterface):
    def __init__(self, venv):
        self.venv = venv
        self._batch_size = venv.num_envs

        # Infer dimensions
        self._obs_space = venv.single_observation_space
        act_space = venv.single_action_space

        self._obs_dim = self._obs_space.n if hasattr(self._obs_space, "n") else self._obs_space.shape[0]
        self._action_dim = act_space.n

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def reset(self, seed=None):
        obs, info = self.venv.reset(seed=seed)

        return {
            "state": torch.tensor(obs, dtype=torch.float32),
            "reward": torch.zeros(self.batch_size),
            "is_first": torch.ones(self.batch_size, dtype=torch.bool),
            "is_last": torch.zeros(self.batch_size, dtype=torch.bool),
            "is_terminal": torch.zeros(self.batch_size, dtype=torch.bool),
            "info": info,
        }

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.venv.step(actions.cpu().numpy())

        is_last = torch.tensor(terminated | truncated, dtype=torch.bool)
        is_terminal = torch.tensor(terminated, dtype=torch.bool)

        return {
            "state": torch.tensor(obs, dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "is_first": torch.zeros(self.batch_size, dtype=torch.bool),
            "is_last": is_last,
            "is_terminal": is_terminal,
            "info": info,
        }
