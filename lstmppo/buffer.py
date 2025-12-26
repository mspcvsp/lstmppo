import torch
from dataclasses import dataclass

@dataclass
class RolloutStep:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    hxs: torch.Tensor
    cxs: torch.Tensor


@dataclass
class RecurrentBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    hxs: torch.Tensor
    cxs: torch.Tensor


class RecurrentRolloutBuffer:

    def __init__(self, cfg):

        self.T = cfg.rollout_steps
        self.N = cfg.num_envs
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.lam = cfg.gae_lambda

        self.obs = torch.zeros(self.T,
                               self.N,
                               *cfg.obs_shape,
                               device=cfg.device)
        
        self.actions = torch.zeros(self.T,
                                   self.N,
                                   1,
                                   device=cfg.device)

        self.rewards = torch.zeros(self.T,
                                   self.N,
                                   device=cfg.device)

        self.values = torch.zeros(self.T,
                                  self.N,
                                  device=cfg.device)
        
        self.logprobs = torch.zeros(self.T,
                                    self.N, device=cfg.device)

        self.terminated = torch.zeros(self.T,
                                      self.N,
                                      device=cfg.device,
                                      dtype=torch.bool)

        self.truncated = torch.zeros(self.T,
                                     self.N,
                                     device=cfg.device,
                                     dtype=torch.bool)

        # Hidden states at start of each timestep
        self.hxs = torch.zeros(self.T,
                               self.N,
                               cfg.hidden_size,
                               device=cfg.device)

        self.cxs = torch.zeros(self.T,
                               self.N,
                               cfg.hidden_size,
                               device=cfg.device)

        self.step = 0

    def add(self,
            step: RolloutStep):

        t = self.step
        self.obs[t].copy_(step.obs)
        # (N,) -> (N,1)
        self.actions[t].copy_(step.actions.unsqueeze(-1))
        self.rewards[t].copy_(step.rewards)
        self.values[t].copy_(step.values)
        self.logprobs[t].copy_(step.logprobs)
        self.terminated[t].copy_(step.terminated)
        self.truncated[t].copy_(step.truncated)
        self.hxs[t].copy_(step.hxs)
        self.cxs[t].copy_(step.cxs)
        self.step += 1

    def compute_gae(self,
                    last_value):

        T, N = self.T, self.N
        advantages = torch.zeros(T, N, device=self.device)
        last_gae = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            true_terminal = self.terminated[t]
            
            # truncated episodes still bootstrap
            bootstrap = ~true_terminal

            next_value = last_value if t == T - 1 else self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * bootstrap
                - self.values[t]
            )

            last_gae =\
                delta + self.gamma * self.lam * last_gae * bootstrap
            
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = self.values + advantages

    def get_recurrent_minibatches(self, batch_envs):
        """
        Entire sequences per environment:
        yields dict with tensors shaped (T, B, ...)
        """
        N = self.N
        env_indices = torch.randperm(N, device=self.device)

        for start in range(0, N, batch_envs):
            idx = env_indices[start:start + batch_envs]

            yield RecurrentBatch(
                obs=self.obs[:, idx],
                actions=self.actions[:, idx],
                values=self.values[:, idx],
                logprobs=self.logprobs[:, idx],
                returns=self.returns[:, idx],
                advantages=self.advantages[:, idx],
                hxs=self.hxs[0, idx],
                cxs=self.cxs[0, idx],
            )

    def reset(self):
        self.step = 0