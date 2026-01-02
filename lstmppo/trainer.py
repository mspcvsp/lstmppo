# trainer.py
import numpy as np
import torch
import random
from torch import nn
from .env import RecurrentVecEnvWrapper, to_policy_input
from .buffer import RecurrentRolloutBuffer, RolloutStep
from .policy import LSTMPPOPolicy
from .types import PPOConfig, PolicyEvalInput, PolicyEvalOutput, PolicyInput


class LSTMPPOTrainer:

    def __init__(self,
                 cfg:PPOConfig):

        self.cfg = cfg
        self.device = cfg.device
        self.update_idx = 0

        self.env = RecurrentVecEnvWrapper(cfg)
        self.policy = LSTMPPOPolicy(cfg).to(self.device)
        self.buffer = RecurrentRolloutBuffer(cfg)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=cfg.learning_rate,
            eps=1e-5
        )

        self.rollout_steps = cfg.rollout_steps
        self.epochs = cfg.update_epochs
        self.clip_range = cfg.clip_range
        self.vf_coef = cfg.vf_coef
        self.target_kl = cfg.target_kl
        self.max_grad_norm = cfg.max_grad_norm
    
        self.fixed_entropy_coef = cfg.fixed_entropy_coef
        self.anneal_entropy_flag = cfg.anneal_entropy_flag
        self.start_entropy_coef = cfg.start_entropy_coef
        self.end_entropy_coef = cfg.end_entropy_coef
        self.entropy_coef_slope = None
        self.seed = cfg.seed

        self.reset()

    def reset(self):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # persistent env state across rollouts
        self.env_state = self.env.reset(seed=self.seed)

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
                hxs=policy_in.hxs,
                cxs=policy_in.cxs,
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
        
    def update_phase_tbptt(self):

        # ----- Entropy annealing -----
        if self.anneal_entropy_flag:

            entropy_coef =\
                self.entropy_coef_slope * self.update_idx +\
                self.start_entropy_coef
        else:
            entropy_coef = self.fixed_entropy_coef  

        # ----- KL-adaptive clipping -----
        # We adjust clip_range AFTER each epoch based on avg KL
        target_kl = self.cfg.target_kl
        kl_accum = 0.0
        kl_count = 0

        tbptt_steps = self.cfg.tbptt_steps   # e.g., 16

        for _ in range(self.epochs):

            for batch in self.buffer.get_recurrent_minibatches():

                # batch.obs: (T, B, obs_dim)
                # batch.hxs: (B, H)
                # batch.cxs: (B, H)

                T, B, _ = batch.obs.shape

                # Initial hidden state for the first chunk
                hxs = batch.hxs
                cxs = batch.cxs

                # Loop over chunks
                for t0 in range(0, T, tbptt_steps):

                    t1 = min(t0 + tbptt_steps, T)

                    obs_chunk = batch.obs[t0:t1]            # (K, B, obs_dim)
                    actions_chunk = batch.actions[t0:t1]    # (K, B, 1)
                    returns_chunk = batch.returns[t0:t1]    # (K, B)
                    adv_chunk = batch.advantages[t0:t1]     # (K, B)
                    old_logp_chunk = batch.logprobs[t0:t1]  # (K, B)
                    old_values_chunk = batch.values[t0:t1]  # (K, B)

                    # ----- Sequence-aware evaluation -----
                    eval_output =\
                        self.policy.evaluate_actions_sequence(
                            PolicyEvalInput(obs=obs_chunk,
                                            hxs=hxs,
                                            cxs=cxs,
                                            actions=actions_chunk)
                        )

                    # Detach hidden state for next chunk (truncate BPTT)
                    hxs = eval_output.new_hxs.detach()
                    cxs = eval_output.new_cxs.detach()

                    # Flatten K,B → K*B
                    Kc, Bc = eval_output.values.shape
                    values = eval_output.values.view(Kc * Bc)
                    new_logprobs = eval_output.logprobs.view(Kc * Bc)
                    old_logp = old_logp_chunk.view(Kc * Bc)
                    returns = returns_chunk.view(Kc * Bc)
                    advantages = adv_chunk.view(Kc * Bc)
                    old_values = old_values_chunk.view(Kc * Bc)

                    # Normalize advantages per chunk
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(unbiased=False) + 1e-8
                    )

                    # ----- PPO ratio -----
                    ratio = torch.exp(new_logprobs - old_logp)

                    # ----- KL tracking -----
                    approx_kl = (old_logp - new_logprobs).mean()
                    kl_accum += approx_kl.item()
                    kl_count += 1

                    # ----- Clipped surrogate -----
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    ) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # ----- Value function clipping -----
                    value_pred_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_range,
                        self.clip_range,
                    )
                    value_losses = (values - returns).pow(2)
                    
                    value_losses_clipped =\
                        (value_pred_clipped - returns).pow(2)
                    
                    value_loss =\
                        0.5 * torch.max(value_losses,
                                        value_losses_clipped).mean()

                    # ----- Total loss -----
                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        - entropy_coef * eval_output.entropy
                        + eval_output.ar_loss
                        + eval_output.tar_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.policy.parameters(),
                                             self.max_grad_norm)

                    self.optimizer.step()

            # ----- KL-adaptive clipping (after each epoch) -----
            avg_kl = kl_accum / max(kl_count, 1)

            if avg_kl > 2.0 * target_kl:
                self.clip_range *= 0.9
            elif avg_kl < 0.5 * target_kl:
                self.clip_range *= 1.05

            self.clip_range = float(torch.clamp(
                torch.tensor(self.clip_range),
                0.05,
                0.3
            ))

    def train(self, total_updates: int):

        if self.anneal_entropy_flag:

            self.entropy_coef_slope =\
                (self.end_entropy_coef - self.start_entropy_coef) /\
                max(total_updates - 1, 1)

        for self.update_idx in range(total_updates):

            self.rollout_phase()

            self.update_phase()

    def validate_lstm_state_flow(self):

        print("=== LSTM State-Flow Validation ===")

        # Ensure deterministic DropConnect behavior
        self.policy.eval()

        # Run a rollout (num_envs = 1 recommended)
        self.rollout_phase()

        # Get the single batch
        batches = list(self.buffer.get_recurrent_minibatches())
        assert len(batches) == 1, "Use num_envs=1 for validation."
        batch = batches[0]

        T, B, _ = batch.obs.shape
        print(f"T = {T}, B = {B}")

        with torch.no_grad():

            eval_output = \
                self.policy.evaluate_actions_sequence(
                    PolicyEvalInput(obs=batch.obs,
                                    hxs=batch.hxs,
                                    cxs=batch.cxs,
                                    actions=batch.actions)
                )

        stored_values = batch.values
        stored_logprobs = batch.logprobs

        val_diff = (eval_output.values - stored_values).abs()
        logp_diff = (eval_output.logprobs - stored_logprobs).abs()

        print("max |values_rec - stored_values|   :", val_diff.max().item())
        print("max |logprobs_rec - stored_logprobs|:", logp_diff.max().item())

        for t in range(min(T, 5)):
            print(eval_output.logprobs[t,:0])

            print(f"\n[t = {t}]")
            print(" stored value   :", stored_values[t, 0].item())
            print(" recomputed val :", eval_output.values[t, 0].item())
            print(" |diff|         :", val_diff[t, 0].item())
            print(" stored logprob :", stored_logprobs[t, 0].item())
            print(" recomputed logp:", eval_output.logprobs[t, 0].item())
            print(" |diff|         :", logp_diff[t, 0].item())

        print("=== Validation complete ===")

    def trace_hidden_states(self, steps=20):

        self.reset()

        self.policy.eval()
        self.env.reset()

        env_state = self.env_state
        hxs_trace = []
        cxs_trace = []

        for _ in range(steps):
            policy_in = to_policy_input(env_state)
            hxs_trace.append(policy_in.hxs.clone())
            cxs_trace.append(policy_in.cxs.clone())

            actions, _, _ = self.policy.act(policy_in)
            env_state = self.env.step(actions)

        return hxs_trace, cxs_trace

    def run_deterministic_rollout(self, steps=50):

        self.policy.eval()
        self.env.reset()
        self.env.set_initial_lstm_states(self.buffer.get_last_lstm_states())

        obs_list = []
        val_list = []
        logp_list = []

        env_state = self.env_state

        for _ in range(steps):
            policy_in = to_policy_input(env_state)
            actions, logprobs, policy_out = self.policy.act(policy_in)

            obs_list.append(env_state.obs.clone())
            val_list.append(policy_out.values.clone())
            logp_list.append(logprobs.clone())

            env_state = self.env.step(actions)

        return obs_list, val_list, logp_list

    def validate_tbptt(self, K=16):

        self.policy.eval()
        self.rollout_phase()

        batch = next(self.buffer.get_recurrent_minibatches())
        T, B, _ = batch.obs.shape
        print(f"T={T}, B={B}, K={K}")

        # ----- Full sequence -----
        with torch.no_grad():
            full = self.policy.evaluate_actions_sequence(
                PolicyEvalInput(
                    obs=batch.obs,        # (T, B, obs_dim)
                    hxs=batch.hxs,        # (B, H)
                    cxs=batch.cxs,        # (B, H)
                    actions=batch.actions # (T, B, 1) or (T, B)
                )
            )
            full_vals = full.values      # (T, B)
            full_logp = full.logprobs    # (T, B)

        assert full_logp.shape == (T, B), \
            f"full_logp must be (T,B), got {full_logp.shape}"

        print("full_vals shape:", full_vals.shape)
        print("full_logp shape:", full_logp.shape)

        # ----- Chunked sequence -----
        hxs = batch.hxs
        cxs = batch.cxs
        vals_chunks = []
        logp_chunks = []

        with torch.no_grad():
            for t0 in range(0, T, K):
                t1 = min(t0 + K, T)

                out = self.policy.evaluate_actions_sequence(
                    PolicyEvalInput(
                        obs=batch.obs[t0:t1],        # (K, B, obs_dim)
                        hxs=hxs,                     # (B, H)
                        cxs=cxs,                     # (B, H)
                        actions=batch.actions[t0:t1] # (K, B, 1) or (K, B)
                    )
                )

                vals_chunks.append(out.values)    # (K, B)
                logp_chunks.append(out.logprobs)  # (K, B)

                # carry hidden state forward across chunks
                hxs = out.new_hxs.detach()        # (B, H)
                cxs = out.new_cxs.detach()        # (B, H)

        vals_rec = torch.cat(vals_chunks, dim=0)   # (T, B)
        logp_rec = torch.cat(logp_chunks, dim=0)   # (T, B)

        assert logp_rec.shape == (T, B), \
            f"logp_rec must be (T,B), got {logp_rec.shape}"

        print("vals_rec shape:", vals_rec.shape)
        print("logp_rec shape:", logp_rec.shape)

        print("TBPTT value diff:",
              (vals_rec - full_vals).abs().max().item())

        print("TBPTT logprob diff:",
              (logp_rec - full_logp).abs().max().item())

    def assert_rollout_deterministic(self):

        self.policy.eval()
        _, v1, p1 = self.run_deterministic_rollout()
        _, v2, p2 = self.run_deterministic_rollout()

        assert torch.allclose(v1[0], v2[0], atol=1e-6), \
            "Rollout values are not deterministic"

        assert torch.allclose(p1[0], p2[0], atol=1e-6), \
            "Rollout logprobs are not deterministic"


