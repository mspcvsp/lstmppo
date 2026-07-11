from __future__ import annotations

from typing import Dict, List

import torch

from dreamerrl.utils.types import TrainingConfig


class Episode:
    """Stores a single episode for Dreamer-V3."""

    def __init__(self):
        self.data: Dict[str, List[torch.Tensor]] = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
        }

    def add(self, obs, action, reward, done):
        self.data["obs"].append(obs)
        self.data["action"].append(action)
        self.data["reward"].append(reward)
        self.data["done"].append(done)

    def finalize(self) -> Dict[str, torch.Tensor]:
        return {k: torch.stack(v, dim=0) for k, v in self.data.items()}


class ReplayBuffer:
    """
    Dreamer-V3 replay buffer:
      - stores full episodes
      - samples fixed-length sequences
      - supports vectorized envs
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        repro_log=None,
        seed: int = 0,
    ):
        self.repro_log = repro_log
        self.cfg = cfg

        self.capacity = self.cfg.replay_capacity
        self.device = device
        self.seq_len = self.cfg.seq_len

        self.episodes: List[Dict[str, torch.Tensor]] = []
        self.current_eps: List[Episode] = []
        self.num_envs = None

        self.size = 0
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.rng = torch.Generator(device)
        self.rng.manual_seed(seed)

    def _ensure_envs(self, num_envs: int):
        if self.num_envs is None:
            self.num_envs = num_envs
            self.current_eps = [Episode() for _ in range(num_envs)]

    def add(self, obs, action, reward, done):
        """
        obs: (num_envs, obs_dim)
        action: (num_envs,)
        reward: (num_envs,)
        done: (num_envs,)
        """
        num_envs = obs.shape[0]
        self._ensure_envs(num_envs)

        for i in range(num_envs):
            self.current_eps[i].add(
                obs[i].detach().to(self.device),
                action[i].detach().to(self.device),
                reward[i].detach().to(self.device),
                done[i].detach().to(self.device),
            )

            if done[i].item() == 1.0:
                self._finalize_episode(i)

    def _finalize_episode(self, idx: int):
        ep = self.current_eps[idx].finalize()
        self.episodes.append(ep)
        self.current_eps[idx] = Episode()

        # capacity control
        self.size += ep["obs"].shape[0]
        while self.size > self.capacity:
            removed = self.episodes.pop(0)
            self.size -= removed["obs"].shape[0]

    def sample(self, batch_size: int, seed: int | None = None) -> Dict[str, torch.Tensor]:
        assert len(self.episodes) > 0, "Replay buffer is empty"

        if seed is not None:
            self.rng.manual_seed(seed)

        obs_batch = []
        act_batch = []
        rew_batch = []
        done_batch = []

        for _ in range(batch_size):
            idx = int(torch.randint(0, len(self.episodes), (1,), generator=self.rng, device=self.device))
            ep = self.episodes[idx]

            length = ep["obs"].shape[0]

            if length <= self.seq_len:
                start = 0
            else:
                start = int(torch.randint(0, length - self.seq_len, (1,), generator=self.rng, device=self.device))

            end = start + self.seq_len

            if self.cfg.enable_repro_log and self.repro_log is not None:
                self.repro_log.debug(f"SAMPLE_IDX: {idx}, START={start}, END={end}")

            obs_batch.append(ep["obs"][start:end])
            act_batch.append(ep["action"][start:end])
            rew_batch.append(ep["reward"][start:end])
            done_batch.append(ep["done"][start:end])

        return {
            "obs": torch.stack(obs_batch).to(self.device),
            "action": torch.stack(act_batch).to(self.device),
            "reward": torch.stack(rew_batch).to(self.device),
            "done": torch.stack(done_batch).to(self.device),
        }
