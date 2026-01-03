# trainer.py
from pathlib import Path
import numpy as np
import torch
import random
from torch import nn
from torch.distributions.categorical import Categorical

from .env import RecurrentVecEnvWrapper, to_policy_input
from .buffer import RecurrentRolloutBuffer, RolloutStep
from .policy import LSTMPPOPolicy
from .types import Config, PolicyEvalInput, PolicyInput, initialize_config
from .types import RecurrentMiniBatch, PolicyUpdateInfo
from .trainer_state import TrainerState


class LSTMPPOTrainer:

    def __init__(self,
                 cfg: Config,
                 validation_mode: bool = False):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            and cfg.trainer.cuda else "cpu"
        )

        self.state = TrainerState(cfg,
                                  validation_mode=validation_mode)

        self.env = RecurrentVecEnvWrapper(self.state.cfg,
                                          self.device)
        
        self.policy = LSTMPPOPolicy(self.state.cfg).to(self.device)

        self.buffer = RecurrentRolloutBuffer(self.state.cfg, self.device)

        if self.state.validation_mode:

            self.policy.eval()

        self.checkpoint_dir = Path(*[self.state.cfg.log.checkpoint_dir,
                                     self.state.cfg.env.env_id,
                                     self.state.cfg.trainer.exp_name])

        if self.checkpoint_dir.exists() is False:

            self.checkpoint_dir.mkdir(parents=True,
                                      exist_ok=True)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.state.cfg.sched.base_lr,
            eps=1e-5
        )

        self.reset()

    @classmethod
    def for_validation(cls):
        """
        Construct a trainer in validation mode.
        Ensures deterministic behavior and single-env operation.
        """
        cfg = Config()
        cfg = initialize_config(cfg)

        return cls(cfg, validation_mode=True)

    @classmethod
    def from_preset(cls,
                    preset_name: str,
                    **kwargs):

        cfg = Config()

        if preset_name == "cartpole_easy":

            cfg.env.env_id = "popgym-PositionOnlyCartPoleEasy-v0"
            cfg.ppo.target_kl = 0.005
            cfg.sched.start_entropy_coef = 0.1
            cfg.sched.end_entropy_coef = 0.0

        cfg = initialize_config(cfg, **kwargs)

        return cls(cfg)

    def reset(self):

        random.seed(self.state.cfg.trainer.seed)
        np.random.seed(self.state.cfg.trainer.seed)
        torch.manual_seed(self.state.cfg.trainer.seed)

        # persistent env state across rollouts
        self.env_state = self.env.reset(seed=self.state.cfg.trainer.seed)

    def train(self,
              total_updates: int):

        self.state.reset(total_updates)

        with open(self.state.jsonl_file, "w") as self.state.jsonl_fp:

            for self.state.update_idx in range(total_updates):

                self.state.apply_schedules(self.optimizer)

                last_value = self.collect_rollout()

                self.buffer.compute_returns_and_advantages(last_value)

                self.optimize_policy()

                if self.state.should_stop_early():
                    break

                if self.state.should_save_checkpoint():
                    self.save_checkpoint()

    # ---------------------------------------------------------
    # Rollout Phase (unchanged except act() signature)
    # ---------------------------------------------------------
    def collect_rollout(self):

        self.buffer.reset()
        env_state = self.env_state

        # Initialize LSTM states for this rollout
        self.env.set_initial_lstm_states(
            self.buffer.get_last_lstm_states()
        )

        for _ in range(self.state.cfg.trainer.rollout_steps):

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

        return last_value

    def optimize_policy(self):

        self.state.init_stats()

        for _ in range(self.state.cfg.epochs):

            for batch in self.buffer.get_recurrent_minibatches():

                for mb in batch.iter_chunks(
                    self.state.cfg.trainer.tbptt_chunk_len
                    ):

                    self.optimize_chunk(mb)

        self.compute_explained_variance()

        self.state.compute_average_stats()

        self.state.adapt_clip_range()

        self.state.log_metrics()

    def optimize_chunk(self,
                       mb: RecurrentMiniBatch):

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
            + self.state.cfg.vf_coef * value_loss
            - self.state.entropy_coef * entropy
        )

        if self.state.cfg.debug_mode is False:

            loss = (
                loss
                + eval_output.ar_loss
                + eval_output.tar_loss
            )

        grad_norm = self.backward_and_clip(loss)

        self.state.update_stats(
            PolicyUpdateInfo(policy_loss=policy_loss,
                             value_loss=value_loss,
                             entropy=entropy,
                             approx_kl=approx_kl,
                             clip_frac=clip_frac,
                             grad_norm=grad_norm)
        )

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
            ((ratio > 1 + self.state.clip_range) |
             (ratio < 1 - self.state.clip_range)).float() * mask
        ).sum() / mask.sum()

        surr1 = ratio * adv

        surr2 = torch.clamp(ratio,
                            1 - self.state.clip_range,
                            1 + self.state.clip_range) * adv

        policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum()

        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.state.clip_range,
            self.state.clip_range,
        )

        value_loss = 0.5 * torch.max(
            (values - returns).pow(2),
            (value_pred_clipped - returns).pow(2)
        )
        value_loss = (value_loss * mask).sum() / mask.sum()

        return policy_loss, value_loss, approx_kl, clip_frac
    
    def backward_and_clip(self,
                          loss):

        # Disabling set to None reduces memory fragmentation and 
        # speeds up training.
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()

        grad_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5

        nn.utils.clip_grad_norm_(self.policy.parameters(),
                                 self.state.cfg.ppo.max_grad_norm)

        self.optimizer.step()

        return grad_norm

    def compute_explained_variance(self):

        #  ----- Compute EV over the entire rollout  -----
        all_values = self.buffer.values.view(-1)
        all_returns = self.buffer.returns.view(-1)
        all_mask = self.buffer.mask.view(-1)
        valid = all_mask > 0.5

        if valid.sum() == 0:
            self.state.stats["explained_var"] = 0.0
        else:
            ev = explained_variance(all_values[valid],
                                    all_returns[valid])

            self.state.stats["explained_var"] = ev.item()

    def save_checkpoint(self):

        checkpoint_pth =\
            self.checkpoint_dir.joinpath(self.state.cfg.log.run_name +
                                         "_checkpoint_" +
                                         f"{self.state.update_idx}.pt")
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_idx": self.state.update_idx,
            "state": self.state,
        }, checkpoint_pth)

    def validate_tbptt(self, K=16):

        self.policy.eval()
        self.collect_rollout()

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
        self.collect_rollout()

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

        print("max |values_rec - stored_values|   :",
              val_diff.max().item())
        
        print("max |logprobs_rec - stored_logprobs|:",
              logp_diff.max().item())

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

def validate():

    trainer = LSTMPPOTrainer.for_validation()

    trainer.assert_hidden_state_flow()
    trainer.assert_rollout_deterministic()
    trainer.validate_tbptt()
    trainer.validate_lstm_state_flow()