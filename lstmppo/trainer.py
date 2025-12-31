# trainer.py
import torch
from torch import nn
from .env import RecurrentVecEnvWrapper, to_policy_input
from .buffer import RecurrentRolloutBuffer, RolloutStep
from .policy import LSTMPPOPolicy


class LSTMPPOTrainer:

    def __init__(self, cfg, venv):

        self.cfg = cfg
        self.device = cfg.device

        self.env = RecurrentVecEnvWrapper(cfg, venv)
        self.policy = LSTMPPOPolicy(cfg).to(self.device)
        self.buffer = RecurrentRolloutBuffer(cfg)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=cfg.lr,
            eps=1e-5
        )

        self.rollout_steps = cfg.rollout_steps
        self.epochs = cfg.ppo_epochs
        self.clip_range = cfg.clip_range
        self.vf_coef = cfg.vf_coef
        self.ent_coef = cfg.ent_coef
        self.max_grad_norm = cfg.max_grad_norm

        # persistent env state across rollouts
        self.env_state = self.env.reset()

    # ---------------------------------------------------------
    # Rollout Phase (unchanged except act() signature)
    # ---------------------------------------------------------
    def rollout_phase(self):

        self.buffer.reset()
        env_state = self.env_state

        # Initialize LSTM states for this rollout
        self.env.set_initial_lstm_states(
            self.buffer.get_last_lstm_states()
        )

        for _ in range(self.rollout_steps):

            policy_in = to_policy_input(env_state)

            # NEW: act() now takes a PolicyInput dataclass
            actions, logprobs, policy_out = self.policy.act(policy_in)

            next_state = self.env.step(actions)

            self.buffer.add(RolloutStep(
                obs=env_state.obs,
                actions=actions,
                rewards=next_state.rewards,
                values=policy_out.values,
                logprobs=logprobs,
                terminated=next_state.terminated,
                truncated=next_state.truncated,
                hxs=policy_out.new_hxs,
                cxs=policy_out.new_cxs,
            ))

            self.env.update_hidden_states(
                policy_out.new_hxs,
                policy_out.new_cxs,
            )

            env_state = next_state

        self.env_state = env_state

        # Bootstrap value + store final LSTM states
        with torch.no_grad():
            policy_in = to_policy_input(self.env_state)
            _, _, last_policy_out = self.policy.act(policy_in)
            last_value = last_policy_out.values
            self.buffer.store_last_lstm_states(last_policy_out)

        self.buffer.compute_returns_and_advantages(last_value)

    # ---------------------------------------------------------
    # Update Phase (FULLY REFACTORED)
    # ---------------------------------------------------------
    def update_phase(self):

        for _ in range(self.epochs):

            for batch in self.buffer.get_recurrent_minibatches():

                # Shapes:
                # obs: (T, B, obs_dim)
                # actions: (T, B, 1)
                # hxs, cxs: (B, H)

                # Sequence-aware evaluation
                values, new_logprobs, entropy, _, _, ar_loss, tar_loss = \
                    self.policy.evaluate_actions_sequence(
                        obs=batch.obs,
                        hxs=batch.hxs,
                        cxs=batch.cxs,
                        actions=batch.actions,
                    )

                # Flatten for PPO loss
                T, B = values.shape
                values = values.view(T * B)
                new_logprobs = new_logprobs.view(T * B)
                old_logprobs = batch.logprobs.view(T * B)
                returns = batch.returns.view(T * B)
                advantages = batch.advantages.view(T * B)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

                # PPO objective
                ratio = torch.exp(new_logprobs - old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns - values).pow(2).mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy
                    + ar_loss
                    + tar_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

    def train(self, total_updates: int):
        for update in range(total_updates):
            self.rollout_phase()
            self.update_phase()