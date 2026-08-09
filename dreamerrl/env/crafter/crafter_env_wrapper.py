import crafter  # noqa: F401
import gym
import numpy as np
import torch
from gymnasium.spaces import Discrete

import dreamerrl.env.crafter.register_crafter  # noqa: F401


class CrafterEnvWrapper:
    """
    Single‑env Crafter wrapper with Dreamer‑style interface:
      - batched obs (B=1)
      - float32 images in [0, 1]
      - keys: state, reward, is_first, is_last, is_terminal
    """

    def __init__(self, cfg_env):
        # cfg_env.env_id, cfg_env.seed, cfg_env.max_episode_steps, etc.
        self.env_id = getattr(cfg_env, "env_id", "crafter-v1")
        self.seed = getattr(cfg_env, "seed", 0)
        self.max_episode_steps = getattr(cfg_env, "max_episode_steps", 10_000)

        self.env = gym.make(self.env_id)
        self.env.reset(seed=self.seed)

        # Crafter: Box(64, 64, 3, uint8), Discrete(17)
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        else:
            raise TypeError("CrafterEnvWrapper expects a Discrete action space.")

        self._step_count = 0

    def reset(self):
        self._step_count = 0
        obs, info = self.env.reset(seed=self.seed)
        obs = self._process_obs(obs)

        state = torch.from_numpy(obs).unsqueeze(0)  # (1, 64, 64, 3)
        reward = torch.zeros(1, dtype=torch.float32)
        is_first = torch.ones(1, dtype=torch.bool)
        is_last = torch.zeros(1, dtype=torch.bool)
        is_terminal = torch.zeros(1, dtype=torch.bool)

        return {
            "state": state,
            "reward": reward,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def step(self, action: torch.Tensor):
        # -----------------------------------------------------------------------------
        # Why Crafter does NOT require one‑hot encoding (unlike PopGym)
        # -----------------------------------------------------------------------------
        # PopGym’s RepeatPrevious* tasks emit a single categorical integer (e.g., 0–3)
        # that represents a *symbolic cue*. Dreamer cannot learn meaningful geometry
        # from raw integers because “2” is not twice “1” — it is just a label. For
        # PopGym, the observation must be converted into a one‑hot vector so the RSSM
        # receives a stable, differentiable representation.
        #
        # Crafter is fundamentally different:
        #
        #   • Observations are already *continuous vectors* — 64×64×3 RGB images.
        #   • The encoder processes images directly using a CNN.
        #   • The RSSM receives the CNN embedding, not the raw action or category.
        #   • The action space is Discrete(17), but DreamerV3 internally converts
        #     sampled actions into one‑hot vectors during imagination.
        #
        # Therefore:
        #   • Crafter observations do NOT need one‑hot encoding.
        #   • Crafter actions should be returned as integer indices.
        #   • DreamerV3 will one‑hot encode actions internally when needed (e.g.,
        #     imagine_step), keeping the world model interface consistent across
        #     environments.
        #
        # In short: PopGym requires one‑hot because its observations are categorical.
        # Crafter does not, because its observations are already continuous tensors.
        # -----------------------------------------------------------------------------
        self._step_count += 1

        a = int(action.item())
        obs, reward, terminated, truncated, info = self.env.step(a)
        done = terminated or truncated

        obs = self._process_obs(obs)

        state = torch.from_numpy(obs).unsqueeze(0)  # (1, 64, 64, 3)
        reward_t = torch.tensor([reward], dtype=torch.float32)

        is_first = torch.tensor([self._step_count == 0], dtype=torch.bool)
        is_last = torch.tensor([done], dtype=torch.bool)
        is_terminal = torch.tensor([terminated], dtype=torch.bool)

        if done:
            # env will be reset by trainer on next collect
            self._step_count = 0

        return {
            "state": state,
            "reward": reward_t,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # uint8 [0,255] -> float32 [0,1]
        obs = obs.astype(np.float32) / 255.0
        return obs

    @property
    def batch_size(self) -> int:
        return 1
