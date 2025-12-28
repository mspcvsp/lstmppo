# trainer.py
import torch
from torch import nn
from .env import RecurrentVecEnvWrapper, to_policy_input
from .buffer import RecurrentRolloutBuffer, RolloutStep
from .policy import LSTMPPOPolicy


class LSTMPPOTrainer:

    def __init__(self,
                 cfg,
                 venv):

        self.cfg = cfg
        self.device = cfg.device

        self.env = RecurrentVecEnvWrapper(cfg, venv)
        self.policy = LSTMPPOPolicy(cfg).to(self.device)
        self.buffer = RecurrentRolloutBuffer(cfg)

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=cfg.lr,
                                          eps=1e-5)

        self.rollout_steps = cfg.rollout_steps
        self.epochs = cfg.ppo_epochs
        self.clip_range = cfg.clip_range
        self.vf_coef = cfg.vf_coef
        self.ent_coef = cfg.ent_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.env_state = self.env.reset()

    def rollout_phase(self):

        self.buffer.reset()
        env_state = self.env_state

        self.env.set_initial_lstm_states(self.buffer.get_last_lstm_states())

        for _ in range(self.rollout_steps):

            policy_in = to_policy_input(env_state)
            
            policy_out = self.policy.act(
                obs=policy_in.obs,
                hxs=policy_in.hxs,
                cxs=policy_in.cxs,
            )

            actions = policy_out.actions

            # Step env
            next_state = self.env.step(actions)

            # Store (s, a, r, s') transition with LSTM states
            self.buffer.add(RolloutStep(
                obs=env_state.obs,
                actions=actions,
                rewards=next_state.rewards,
                values=policy_out.values,
                logprobs=policy_out.logprobs,
                terminated=next_state.terminated,
                truncated=next_state.truncated,
                hxs=policy_out.new_hxs,
                cxs=policy_out.new_cxs,
            ))

            # Update env hidden states
            self.env.update_hidden_states(
                policy_out.new_hxs,
                policy_out.new_cxs,
            )

            env_state = next_state

        self.env_state = env_state

        # Bootstrap value + store final LSTM states
        with torch.no_grad():

            policy_in = to_policy_input(self.env_state)

            last_policy_out = self.policy.act(
                obs=policy_in.obs,
                hxs=policy_in.hxs,
                cxs=policy_in.cxs,
            )

            last_value = last_policy_out.values
            self.buffer.store_last_lstm_states(last_policy_out)

        self.buffer.compute_returns_and_advantages(last_value=last_value)

    def update_phase(self):
        for _ in range(self.epochs):
            for batch in self.buffer.get_recurrent_minibatches():
                # Flatten time and batch dims for obs / actions
                T, B = batch.obs.shape[0], batch.obs.shape[1]
                obs = batch.obs.view(T * B, -1)
                actions = batch.actions.view(T * B, -1)
                old_logprobs = batch.logprobs.view(T * B)
                returns = batch.returns.view(T * B)
                advantages = batch.advantages.view(T * B)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

                # Evaluate policy on the sequence
                # NOTE: This is a simple "flattened" LSTM call; for full
                # sequence-aware LSTM, you’d feed (T, B, ...) and carry hxs across time.
                values, logprobs, entropy, _, _ = self.policy.evaluate_actions(
                    obs=obs,
                    hxs=batch.hxs,  # this is not fully sequence-aware yet
                    cxs=batch.cxs,
                    actions=actions,
                )

                # PPO objective
                ratio = torch.exp(logprobs - old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

    def train(self, total_updates: int):
        for update in range(total_updates):
            self.rollout_phase()
            self.update_phase()