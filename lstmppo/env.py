import gymnasium as gym
from dataclasses import dataclass
from typing import Any, List
import torch


@dataclass
class VecEnvState:
    obs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: List[Any]
    hxs: torch.Tensor
    cxs: torch.Tensor


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        return env
    return thunk

class RecurrentVecEnvWrapper:
    """
    Wrap vectorized env and manage per-env LSTM hidden states.
    """

    def __init__(self,
                 cfg,
                 venv):

        self.venv = venv
        self.num_envs = venv.num_envs
        self.hidden_size = cfg.hidden_size
        self.device = cfg.device

        self.hxs = torch.zeros(self.num_envs,
                               cfg.hidden_size,
                               device=cfg.device)

        self.cxs = torch.zeros(self.num_envs,
                               cfg.hidden_size,
                               device=cfg.device)

    def reset(self):

        obs, info = self.venv.reset()
        self.hxs.zero_()
        self.cxs.zero_()

        obs = torch.as_tensor(obs,
                              device=self.device,
                              dtype=torch.float32)

        return VecEnvState(
            obs=obs,
            rewards=torch.zeros(self.num_envs,
                                device=self.device),
            terminated=torch.zeros(self.num_envs,
                                   dtype=torch.bool,
                                   device=self.device),
            truncated=torch.zeros(self.num_envs,
                                  dtype=torch.bool,
                                  device=self.device),
            info=info,
            hxs=self.hxs.clone(),
            cxs=self.cxs.clone(),
        )


    def step(self, actions):
        """
        actions: (N,) int64 tensor
        """
        obs, rewards, terminated, truncated, info = self.venv.step(
            actions.cpu().numpy()
        )

        obs = torch.as_tensor(obs,
                              device=self.device,
                              dtype=torch.float32)
        
        rewards = torch.as_tensor(rewards,
                                  device=self.device,
                                  dtype=torch.float32)

        terminated = torch.as_tensor(terminated,
                                     device=self.device,
                                     dtype=torch.bool)
        
        truncated = torch.as_tensor(truncated,
                                    device=self.device,
                                    dtype=torch.bool)

        # Reset hidden states only for true terminals
        done_mask = terminated
        if done_mask.any():
            self.hxs[done_mask] = 0.0
            self.cxs[done_mask] = 0.0

        return VecEnvState(
            obs=obs,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
            hxs=self.hxs.clone(),
            cxs=self.cxs.clone(),
        )

    def update_hidden_states(self, new_hxs, new_cxs):
        self.hxs.copy_(new_hxs)
        self.cxs.copy_(new_cxs)