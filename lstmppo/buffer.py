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

        # --- Storage ---
        self.obs = torch.zeros(
            self.rollout_steps, self.num_envs, *cfg.obs_shape, device=self.device
        )

        self.actions = torch.zeros(
            self.rollout_steps, self.num_envs, 1, device=self.device
        )

        self.rewards = torch.zeros(
            self.rollout_steps, self.num_envs, device=self.device
        )

        self.values = torch.zeros(
            self.rollout_steps, self.num_envs, device=self.device
        )

        self.logprobs = torch.zeros(
            self.rollout_steps, self.num_envs, device=self.device
        )

        self.terminated = torch.zeros(
            self.rollout_steps, self.num_envs, dtype=torch.bool, device=self.device
        )

        self.truncated = torch.zeros(
            self.rollout_steps, self.num_envs, dtype=torch.bool, device=self.device
        )

        # Hidden states at *start* of each timestep
        self.hxs = torch.zeros(
            self.rollout_steps, self.num_envs, self.lstm_hidden_size, device=self.device
        )
        self.cxs = torch.zeros(
            self.rollout_steps, self.num_envs, self.lstm_hidden_size, device=self.device
        )

        self.returns = torch.zeros(
            self.rollout_steps, self.num_envs, device=self.device
        )

        self.advantages = torch.zeros(
            self.rollout_steps, self.num_envs, device=self.device
        )

        self.reset()

    # ---------------------------------------------------------
    # Add rollout step
    # ---------------------------------------------------------
    def add(self, step: RolloutStep):

        t = self.step

        # --- Shape checks ---
        assert step.obs.shape == (self.num_envs, *self.obs.shape[2:]), \
            f"Obs shape mismatch: {step.obs.shape}"

        assert step.hxs.shape == (self.num_envs, self.lstm_hidden_size)
        assert step.cxs.shape == (self.num_envs, self.lstm_hidden_size)

        # --- Store rollout data ---
        self.obs[t].copy_(step.obs)

        # Actions must be (B,1)
        if step.actions.dim() == 1:
            self.actions[t].copy_(step.actions.unsqueeze(-1))
        elif step.actions.dim() == 2 and step.actions.size(-1) == 1:
            self.actions[t].copy_(step.actions)
        else:
            raise RuntimeError(f"Invalid action shape: {step.actions.shape}")

        self.rewards[t].copy_(step.rewards)
        self.values[t].copy_(step.values)
        self.logprobs[t].copy_(step.logprobs)
        self.terminated[t].copy_(step.terminated)
        self.truncated[t].copy_(step.truncated)

        # Hidden state at *start* of timestep t
        self.hxs[t].copy_(step.hxs)
        self.cxs[t].copy_(step.cxs)

        self.step += 1

    # ---------------------------------------------------------
    # Store final LSTM states for next rollout
    # ---------------------------------------------------------
    def store_last_lstm_states(self, last_policy_output):

        self.last_hxs = last_policy_output.new_hxs.detach()
        self.last_cxs = last_policy_output.new_cxs.detach()

    # ---------------------------------------------------------
    # GAE-Lambda
    # ---------------------------------------------------------
    def compute_returns_and_advantages(self, last_value):

        last_gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.rollout_steps)):

            true_terminal = self.terminated[t]

            # truncated episodes bootstrap; terminated do not
            bootstrap = ~true_terminal

            next_value = (
                last_value if t == self.rollout_steps - 1
                else self.values[t + 1]
            )

            delta = (
                self.rewards[t]
                + self.gamma * next_value * bootstrap
                - self.values[t]
            )

            last_gae = delta + self.gamma * self.lam * last_gae * bootstrap
            self.advantages[t] = last_gae

        self.returns = self.values + self.advantages

    # ---------------------------------------------------------
    # Yield minibatches of full sequences (T, B, ...)
    # ---------------------------------------------------------
    def get_recurrent_minibatches(self):

        env_indices = torch.randperm(self.num_envs, device=self.device)

        for start in range(0, self.num_envs, self.mini_batch_envs):

            idx = env_indices[start:start + self.mini_batch_envs]

            # Mask: 1 = valid, 0 = terminated/truncated at that timestep
            mask =\
                1.0 - (self.terminated[:, idx] |
                       self.truncated[:, idx]).float()

            yield RecurrentBatch(
                obs=self.obs[:, idx],
                actions=self.actions[:, idx],
                values=self.values[:, idx],
                logprobs=self.logprobs[:, idx],
                returns=self.returns[:, idx],
                advantages=self.advantages[:, idx],
                hxs=self.hxs[:, idx],
                cxs=self.cxs[:, idx],
                terminated=self.terminated[:, idx],
                truncated=self.truncated[:, idx],
                mask=mask,
            )

    # ---------------------------------------------------------
    # LSTM state handoff to env wrapper
    # ---------------------------------------------------------
    def get_last_lstm_states(self):

        return LSTMStates(
            hxs=self.last_hxs,
            cxs=self.last_cxs
        )

    # ---------------------------------------------------------
    # Reset buffer
    # ---------------------------------------------------------
    def reset(self):

        self.step = 0
        self.last_hxs = None
        self.last_cxs = None

    # ---------------------------------------------------------
    # Optional safety check
    # ---------------------------------------------------------
    def finalize(self):
        assert self.step == self.rollout_steps, \
            f"Rollout incomplete: {self.step}/{self.rollout_steps}"