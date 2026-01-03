# trainer.py
import json
from pathlib import Path
import numpy as np
import torch
import random
from torch import nn
from torch.distributions.categorical import Categorical
from .env import RecurrentVecEnvWrapper, to_policy_input
from .buffer import RecurrentRolloutBuffer, RolloutStep
from .policy import LSTMPPOPolicy
from .types import PPOConfig, PolicyEvalInput, PolicyInput
from .types import RecurrentMiniBatch
from torch.utils.tensorboard import SummaryWriter


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

        self.entropy_coef_slope = None
        self.jsonl_fp = None

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
        self.seed = cfg.seed
        self.tbptt_steps = cfg.tbptt_steps
        self.stats = None
        self.stat_steps = None

        tb_logdir = Path(*[cfg.tb_logdir,
                           cfg.run_name])

        self.writer = SummaryWriter(log_dir=tb_logdir)

        self.jsonl_file = Path(*[cfg.jsonl_path,
                               cfg.run_name + ".json"])

        self.reset()

    def reset(self):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # persistent env state across rollouts
        self.env_state = self.env.reset(seed=self.seed)

    def train(self,
              total_updates: int):

        if self.anneal_entropy_flag:

            self.entropy_coef_slope =\
                (self.end_entropy_coef - self.start_entropy_coef) /\
                max(total_updates - 1, 1)

        with open(self.jsonl_file, "w") as self.jsonl_fp:

            for self.update_idx in range(total_updates):

                self.rollout_phase()

                self.update_phase_tbptt()

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

        entropy_coef = self.compute_entropy_coef()

        self.init_stats()

        for _ in range(self.epochs):

            for batch in self.buffer.get_recurrent_minibatches():

                for mb in batch.iter_chunks(self.cfg.tbptt_steps):

                    self.update_chunk(mb, entropy_coef)

        #  ----- Compute EV over the entire rollout  -----
        all_values = self.buffer.values.view(-1)
        all_returns = self.buffer.returns.view(-1)
        all_mask = self.buffer.mask.view(-1)
        valid = all_mask > 0.5

        if valid.sum() == 0:
            self.stats["explained_var"] = 0.0
        else:
            ev = explained_variance(all_values[valid],
                                    all_returns[valid])
            
            self.stats["explained_var"] = ev.item()

        self.compute_average_stats()

        self.adapt_clip_range()

        self.log_metrics()

    def update_chunk(self, mb: RecurrentMiniBatch, entropy_coef):

        # ------- Forward pass -------
        eval_output = self.policy.evaluate_actions_sequence(
            PolicyEvalInput(
                obs=mb.obs,
                hxs=mb.hxs0,
                cxs=mb.cxs0,
                actions=mb.actions,
            )
        )

        # ------- Flatten -------
        K, B = eval_output.values.shape
        values = eval_output.values.view(K * B)
        new_logp = eval_output.logprobs.view(K * B)
        old_logp = mb.old_logp.view(K * B)
        returns = mb.returns.view(K * B)
        adv = mb.advantages.view(K * B)
        old_values = mb.old_values.view(K * B)
        mask = mb.mask.view(K * B) # (K*B,)

        if mask.sum() == 0:
            # All envs terminated/truncated at this chunk.
            # No valid timesteps → skip this chunk entirely.
            return

        valid_adv = adv[mask > 0.5]
        
        adv =\
            (adv - valid_adv.mean()) /\
            (valid_adv.std(unbiased=False) + 1e-8)

        policy_loss, value_loss, approx_kl, clip_frac = \
            self.compute_losses(values,
                                new_logp,
                                old_logp,
                                old_values,
                                returns,
                                adv,
                                mask)

        # --- Masked entropy from eval_output ---
        entropy = eval_output.entropy.view(K * B)   # (K*B,)
        entropy = (entropy * mask).sum() / mask.sum()

        loss = (
            policy_loss
            + self.vf_coef * value_loss
            - entropy_coef * entropy
            + eval_output.ar_loss
            + eval_output.tar_loss
        )

        grad_norm = self.backward_and_clip(loss)

        self.update_stats(policy_loss,
                            value_loss,
                            entropy,
                            approx_kl,
                            clip_frac,
                            grad_norm)

    def compute_losses(self,
                       values,
                       new_logp,
                       old_logp,
                       old_values,
                       returns,
                       adv,
                       mask):

        ratio = torch.exp(new_logp - old_logp)

        kl = 0.5 * (old_logp - new_logp).pow(2)
        approx_kl = (kl * mask).sum() / mask.sum()

        clip_frac = (
            ((ratio > 1 + self.clip_range) |
             (ratio < 1 - self.clip_range)).float() * mask
        ).sum() / mask.sum()

        surr1 = ratio * adv

        surr2 = torch.clamp(ratio,
                            1 - self.clip_range,
                            1 + self.clip_range) * adv

        policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum()

        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_range,
            self.clip_range,
        )

        value_loss = 0.5 * torch.max(
            (values - returns).pow(2),
            (value_pred_clipped - returns).pow(2)
        )
        value_loss = (value_loss * mask).sum() / mask.sum()

        return policy_loss, value_loss, approx_kl, clip_frac
    
    def backward_and_clip(self,
                          loss):

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5

        nn.utils.clip_grad_norm_(self.policy.parameters(),
                                 self.max_grad_norm)
        
        self.optimizer.step()

        return grad_norm

    def update_stats(self,
                     policy_loss,
                     value_loss,
                     entropy,
                     approx_kl,
                     clip_frac,
                     grad_norm):

        self.stats["policy_loss"] += policy_loss.item()
        self.stats["value_loss"] += value_loss.item()
        self.stats["entropy"] += entropy.item()
        self.stats["approx_kl"] += approx_kl.item()
        self.stats["clip_frac"] += clip_frac.item()
        self.stats["grad_norm"] += grad_norm
        self.stats["steps"] += 1

    def compute_average_stats(self):

        if self.stats["steps"] > 0:

            norm_factor = 1.0 / self.stats["steps"]

            for key in ["policy_loss",
                        "value_loss",
                        "entropy",
                        "approx_kl",
                        "clip_frac",
                        "grad_norm"]:

                self.stats[key] *= norm_factor

    def adapt_clip_range(self):

        avg_kl = self.stats["approx_kl"]

        if avg_kl > 2.0 * self.target_kl:
            self.clip_range *= 0.9
        elif avg_kl < 0.5 * self.target_kl:
            self.clip_range *= 1.05

        self.clip_range = float(torch.clamp(
            torch.tensor(self.clip_range),
            0.05,
            0.3
        ))

    def log_metrics(self):

        for key, value in self.stats.items():

            self.writer.add_scalar(key,
                                   value,
                                   self.update_idx)

        record = {"update": self.update_idx, **self.stats}
        self.jsonl_fp.write(json.dumps(record) + "\n")
        self.jsonl_fp.flush()

        print(
            f"upd {self.update_idx:04d} | "
            f"pol {self.stats['policy_loss']:.3f} | "
            f"val {self.stats['value_loss']:.3f} | "
            f"ent {self.stats['entropy']:.3f} | "
            f"kl {self.stats['approx_kl']:.4f} | "
            f"clip {self.stats['clip_frac']:.3f} | "
            f"ev {self.stats['explained_var']:.3f} | "
            f"grad {self.stats['grad_norm']:.2f} | "
            f"clip_range {self.clip_range:.3f}"
        )

    def compute_entropy_coef(self):

        if self.anneal_entropy_flag:

            entropy_coef =\
                self.entropy_coef_slope * self.update_idx +\
                self.start_entropy_coef
        else:
            entropy_coef = self.fixed_entropy_coef

        return entropy_coef 
    
    def init_stats(self):

        self.stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
            "explained_var": 0.0,
            "steps": 0,
        }

    def validate_tbptt(self, K=16):
        self.policy.eval()
        self.rollout_phase()

        batch = next(self.buffer.get_recurrent_minibatches())
        T, B, _ = batch.obs.shape

        # Use the hidden state at the start of the sequence
        hxs0 = batch.hxs[0]   # (B, H)
        cxs0 = batch.cxs[0]   # (B, H)

        # ----- Full sequence -----
        with torch.no_grad():
            full = self.policy.evaluate_actions_sequence(
                PolicyEvalInput(
                    obs=batch.obs,        # (T, B, obs_dim)
                    hxs=hxs0,             # (B, H)
                    cxs=cxs0,             # (B, H)
                    actions=batch.actions # (T, B, 1) or (T, B)
                )
            )
            full_vals = full.values      # (T, B)
            full_logp = full.logprobs    # (T, B)

        # ----- Chunked sequence -----
        hxs = hxs0.clone()
        cxs = cxs0.clone()
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
                        actions=batch.actions[t0:t1] # (K, B)
                    )
                )

                vals_chunks.append(out.values)
                logp_chunks.append(out.logprobs)

                hxs = out.new_hxs.detach()
                cxs = out.new_cxs.detach()

        vals_rec = torch.cat(vals_chunks, dim=0)
        logp_rec = torch.cat(logp_chunks, dim=0)

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
        
    def assert_hidden_state_flow(self):

        h1, c1 = self.trace_hidden_states()
        h2, c2 = self.trace_hidden_states()

        for t in range(len(h1)):
            assert torch.allclose(h1[t], h2[t], atol=1e-6), \
                f"hxs mismatch at t={t}"
            assert torch.allclose(c1[t], c2[t], atol=1e-6), \
                f"cxs mismatch at t={t}"

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
    
    def validate_lstm_state_flow(self):
        print("=== LSTM State-Flow Validation ===")

        # Deterministic policy
        self.policy.eval()

        # Single-env rollout recommended
        self.rollout_phase()

        batches = list(self.buffer.get_recurrent_minibatches())
        assert len(batches) == 1, "Use num_envs=1 for validation."
        batch = batches[0]

        T, B, _ = batch.obs.shape
        print(f"T = {T}, B = {B}")

        stored_values = batch.values          # (T, B)
        stored_logprobs = batch.logprobs      # (T, B)

        rec_values = []
        rec_logprobs = []

        with torch.no_grad():
            for t in range(T):
                obs_t = batch.obs[t]          # (B, obs_dim)
                hxs_t = batch.hxs[t]          # (B, H)
                cxs_t = batch.cxs[t]          # (B, H)
                act_t = batch.actions[t]      # (B, 1) or (B,)

                # Build single-step PolicyInput
                policy_in = PolicyInput(
                    obs=obs_t,                # (B, obs_dim)
                    hxs=hxs_t,                # (B, H)
                    cxs=cxs_t,                # (B, H)
                )

                policy_out = self.policy.forward(policy_in)
                dist = Categorical(logits=policy_out.logits)

                actions = act_t.squeeze(-1) if act_t.dim() == 2 else act_t

                val_t = policy_out.values          # (B,)
                logp_t = dist.log_prob(actions)    # (B,)

                rec_values.append(val_t.unsqueeze(0))    # (1, B)
                rec_logprobs.append(logp_t.unsqueeze(0)) # (1, B)

        values_rec = torch.cat(rec_values, dim=0)      # (T, B)
        logprobs_rec = torch.cat(rec_logprobs, dim=0)  # (T, B)

        val_diff = (values_rec - stored_values).abs()
        logp_diff = (logprobs_rec - stored_logprobs).abs()

        print("max |values_rec - stored_values|   :", val_diff.max().item())
        print("max |logprobs_rec - stored_logprobs|:", logp_diff.max().item())

        for t in range(min(T, 5)):
            print(f"\n[t = {t}]")
            print(" stored value   :", stored_values[t, 0].item())
            print(" recomputed val :", values_rec[t, 0].item())
            print(" |diff|         :", val_diff[t, 0].item())
            print(" stored logprob :", stored_logprobs[t, 0].item())
            print(" recomputed logp:", logprobs_rec[t, 0].item())
            print(" |diff|         :", logp_diff[t, 0].item())

        print("=== Validation complete ===")

def explained_variance(y_pred, y_true):
    var_y = torch.var(y_true)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)
