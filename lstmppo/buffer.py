import torch
from .types import RolloutStep, RecurrentBatch, LSTMStates


class RecurrentRolloutBuffer:

    def __init__(self, cfg):

        self.rollout_steps = cfg.rollout_steps
        self.num_envs = cfg.num_envs
        self.mini_batch_envs = cfg.mini_batch_envs
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.lam = cfg.gae_lambda
        self.lstm_hidden_size = cfg.lstm_hidden_size
        self.mini_batch_envs = cfg.mini_batch_envs

        self.last_hxs = None
        self.last_cxs = None

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
                               self.lstm_hidden_size,
                               device=cfg.device)

        self.cxs = torch.zeros(self.rollout_steps,
                               self.num_envs,
                               self.lstm_hidden_size,
                               device=cfg.device)

        self.returns = torch.zeros(self.rollout_steps,
                                   self.num_envs,
                                   device=self.device)
    
        self.advantages = torch.zeros(self.rollout_steps,
                                      self.num_envs,
                                      device=self.device)

        self.reset()

    def add(self,
            step: RolloutStep):

        t = self.step

        assert step.obs.shape == (self.num_envs, *self.obs.shape[2:]), \
            f"Obs shape mismatch: {step.obs.shape}"

        self.obs[t].copy_(step.obs)

        # PPO expects integer action indices of shape (T, B)
        # (or (T, B, 1) before squeeze).
        if step.actions.dim() == 1:
            # (B,) → (B,1)
            self.actions[t].copy_(step.actions.unsqueeze(-1))
        #------------------------------------------------------
        elif step.actions.dim() == 2 and step.actions.size(-1) == 1:
            # already (B,1)
            self.actions[t].copy_(step.actions)
        #------------------------------------------------------
        else:
            raise RuntimeError(f"Invalid action shape: {step.actions.shape}")

        assert self.actions[t].shape == (self.num_envs, 1), \
            f"Stored actions must be (B,1), got {self.actions[t].shape}"

        self.rewards[t].copy_(step.rewards)
        self.values[t].copy_(step.values)
        self.logprobs[t].copy_(step.logprobs)
        self.terminated[t].copy_(step.terminated)
        self.truncated[t].copy_(step.truncated)

        # hidden state at *start* of timestep t
        self.hxs[t].copy_(step.hxs)
        self.cxs[t].copy_(step.cxs)
        self.step += 1

    def store_last_lstm_states(self,
                               last_policy_output):

        """
        Prevent PyTorch from trying to backpropagate through
        the entire rollout when these states are used.
        """
        self.last_hxs = last_policy_output.new_hxs.detach()
        self.last_cxs = last_policy_output.new_cxs.detach()

    def compute_returns_and_advantages(self,
                                       last_value):

        last_gae = torch.zeros(self.num_envs,
                               device=self.device)

        for t in reversed(range(self.rollout_steps)):

            true_terminal = self.terminated[t]
            
            # truncated episodes still bootstrap
            bootstrap = ~true_terminal

            next_value =\
                last_value if t == self.rollout_steps - 1\
                else self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * bootstrap
                - self.values[t]
            )

            last_gae =\
                delta + self.gamma * self.lam * last_gae * bootstrap
            
            self.advantages[t] = last_gae

        self.returns = self.values + self.advantages

    def get_recurrent_minibatches(self):
        """
        Entire sequences per environment:
        yields dict with tensors shaped (rollout_step, B, ...)
        """
        env_indices = torch.randperm(self.num_envs,
                                     device=self.device)

        for start in range(0, self.num_envs, self.mini_batch_envs):

            idx = env_indices[start:start + self.mini_batch_envs]

            yield RecurrentBatch(
                obs=self.obs[:, idx],
                actions=self.actions[:, idx],
                values=self.values[:, idx],
                logprobs=self.logprobs[:, idx],
                returns=self.returns[:, idx],
                advantages=self.advantages[:, idx],
                hxs=self.hxs[0, idx],
                cxs=self.cxs[0, idx],
                terminated=self.terminated[:, idx],
                truncated=self.truncated[:, idx],
            )

    def get_last_lstm_states(self):
        """
        environment wrapper handles the “None means first rollout” case
    
        :param self: RecurrentRolloutBuffer instance
        :return: LSTMStates dataclass with last_hxs and last_cxs tensors
        """
        return LSTMStates(
            hxs=self.last_hxs,
            cxs=self.last_cxs
        )                

    def reset(self):
        self.step = 0