import torch
from .types import RolloutStep, RecurrentBatch


class RecurrentRolloutBuffer:

    def __init__(self, cfg):

        self.rollout_steps = cfg.rollout_steps
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.lam = cfg.gae_lambda

        self.obs = torch.zeros(self.rollout_steps,
                               self.num_envs,
                               *cfg.obs_shape,
                               device=cfg.device)
        
        self.actions = torch.zeros(self.rollout_steps,
                                   self.num_envs,
                                   1,
                                   device=cfg.device)

        self.rewards = torch.zeros(self.rollout_steps,
                                   self.num_envs,
                                   device=cfg.device)

        self.values = torch.zeros(self.rollout_steps,
                                  self.num_envs,
                                  device=cfg.device)
        
        self.logprobs = torch.zeros(self.rollout_steps,
                                    self.num_envs,
                                    device=cfg.device)

        self.terminated = torch.zeros(self.rollout_steps,
                                      self.num_envs,
                                      device=cfg.device,
                                      dtype=torch.bool)

        self.truncated = torch.zeros(self.rollout_steps,
                                     self.num_envs,
                                     device=cfg.device,
                                     dtype=torch.bool)

        # Hidden states at start of each timestep
        self.hxs = torch.zeros(self.rollout_steps,
                               self.num_envs,
                               cfg.lstm_hidden_size,
                               device=cfg.device)

        self.cxs = torch.zeros(self.rollout_steps,
                               self.num_envs,
                               cfg.lstm_hidden_size,
                               device=cfg.device)

        self.step = 0

    def add(self,
            step: RolloutStep):

        t = self.step
        self.obs[t].copy_(step.obs)

        # (self.num_envs,) -> (self.num_envs,1)
        self.actions[t].copy_(step.actions.unsqueeze(-1))
        self.rewards[t].copy_(step.rewards)
        self.values[t].copy_(step.values)
        self.logprobs[t].copy_(step.logprobs)
        self.terminated[t].copy_(step.terminated)
        self.truncated[t].copy_(step.truncated)
        self.hxs[t].copy_(step.hxs)
        self.cxs[t].copy_(step.cxs)
        self.step += 1

    def compute_returns_and_advantages(self,
                                       last_value):

        advantages = torch.zeros(self.rollout_steps,
                                 self.num_envs,
                                 device=self.device)

        last_gae = torch.zeros(self.num_envs,
                               device=self.device)

        for t in reversed(range(self.rollout_step)):
            true_terminal = self.terminated[t]
            
            # truncated episodes still bootstrap
            bootstrap = ~true_terminal

            next_value =\
                last_value if t == self.rollout_step - 1\
                else self.values[t + 1]

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

    def get_recurrent_minibatches(self,
                                  batch_envs):
        """
        Entire sequences per environment:
        yields dict with tensors shaped (rollout_step, B, ...)
        """
        self.num_envs = self.num_envs

        env_indices = torch.randperm(self.num_envs,
                                     device=self.device)

        for start in range(0, self.num_envs, batch_envs):

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