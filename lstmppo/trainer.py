"""
Environment wrapper
• 	raw reward
• 	shaping bonus (terminal only)
• 	return shaped reward

Rollout buffer
• 	store shaped reward
• 	normalize rewards
• 	compute returns
• 	compute advantages

Trainer
• 	normalize advantages
• 	compute PPO losses
• 	update policy

LSTM-PPO pipeline
-----------------
Env → shaped reward
      ↓
Buffer stores rewards
      ↓
Reward normalization
      ↓
GAE computes advantages
      ↓
Value-target whitening (normalize returns)
      ↓
Advantage normalization (trainer)
      ↓
PPO loss
"""
from pathlib import Path
import numpy as np
import torch
from typing import Optional
from types import SimpleNamespace
import random
from torch import nn
from torch.distributions.categorical import Categorical
from rich.live import Live
from rich.console import Console

from .env import RecurrentVecEnvWrapper
from .buffer import RecurrentRolloutBuffer, RolloutStep
from .policy import LSTMPPOPolicy
from .types import Config, PolicyEvalInput, PolicyInput, initialize_config
from .types import RecurrentMiniBatch, PolicyUpdateInfo, PolicyEvalOutput
from .types import LSTMUnitDiagnostics, LSTMGateEntropy, LSTMGateSaturation
from .types import LSTMGates
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

        self.buffer = RecurrentRolloutBuffer(self.state.cfg,
                                             self.device)

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

    @property
    def num_envs(self) -> int:
        return self.state.cfg.env.num_envs

    @property
    def rollout_steps(self) -> int:
        return self.state.cfg.trainer.rollout_steps

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

        self.state.init_stats()

        console = Console()

        with open(self.state.jsonl_file, "w") as self.state.jsonl_fp,\
            Live(console=console,
                 refresh_per_second=4) as live:

            for self.state.update_idx in range(total_updates):

                self.state.init_stats()

                self.state.apply_schedules(self.optimizer)

                last_value = self.collect_rollout()

                self.buffer.compute_returns_and_advantages(last_value)

                self.optimize_policy()

                live.update(self.state.render_dashboard())

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

        for _ in range(self.rollout_steps):

            policy_in = env_state.policy_input

            # NEW: act() now takes a PolicyInput dataclass
            with torch.no_grad():
                actions, logprobs, policy_out = self.policy.act(policy_in)

            next_state = self.env.step(actions)

            """
            During rollout, policy.act() is single‑step, so:
            - policy_out.new_hxs is (B, H)
            - NOT (B, T, H)
            """
            hxs = policy_out.new_hxs.detach()
            cxs = policy_out.new_cxs.detach()

            self.buffer.add(RolloutStep(
                obs=env_state.obs,
                actions=actions.detach(),
                rewards=next_state.rewards,
                values=policy_out.values.detach(),
                logprobs=logprobs.detach(),
                terminated=next_state.terminated,
                truncated=next_state.truncated,
                hxs=hxs,
                cxs=cxs,
                gates=policy_out.gates.detached.transposed()
            ))

            """
            Detach to prevent
            -----------------
            - the autograd graph spans the entire rollout
            - memory exploding
            - PPO becoming unstable
            - gradients flowing across episode boundarie
            """
            self.env.update_hidden_states(hxs,
                                          cxs)

            env_state = next_state

        self.env_state = env_state

        # Bootstrap value + store final LSTM states
        with torch.no_grad():

            policy_in = self.env_state.policy_input

            _, _, last_policy_out = self.policy.act(policy_in)

            """
            Avoids GAE shape mismatches if policy returns (B,1) instead of
            (B,)
            """
            last_value = last_policy_out.values.squeeze(0).detach()

            self.buffer.store_last_lstm_states(last_policy_out)

        ep_stats = self.env.get_episode_stats()

        self.state.update_episode_stats(ep_stats)

        return last_value

    def optimize_policy(self):

        for _ in range(self.state.cfg.ppo.update_epochs):

            for batch in self.buffer.get_recurrent_minibatches():

                for mb in batch.iter_chunks(
                    self.state.cfg.trainer.tbptt_chunk_len
                    ):

                    self.optimize_chunk(mb)

        self.state.compute_explained_variance(self.buffer)

        self.state.compute_average_metrics()

        self.state.adapt_clip_range()

        self.state.kl_watchdog()

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
        # eval_output.values: [K, B] or [K, B, 1]
        values = eval_output.values

        if values.dim() == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)

        values = values.reshape(-1)  # safer than view(K * B)

        new_logp = eval_output.logprobs.reshape(-1)
        old_logp = mb.old_logp.reshape(-1)
        adv = mb.advantages.reshape(-1)
        returns = mb.returns.reshape(-1)
        old_values = mb.old_values.reshape(-1)
    
        """
        Need two views of the mask:

        - Flattened mask for scalar losses: shape (K * B,)
        - Time–batch mask for diagnostics: shape (K, B)
        """
        mask_tb = mb.mask # (K, B)
        mask_flat = mask_tb.reshape(-1)

        if mask_flat.sum() == 0:
            # All envs terminated/truncated at this chunk.
            # No valid timesteps → skip this chunk entirely.
            return

        valid_adv = adv[mask_flat > 0.5]
        
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
                                mask_flat)

        # --- Masked entropy from eval_output ---
        entropy = eval_output.entropy.reshape(-1)
        entropy = (entropy * mask_flat).sum() / mask_flat.sum()

        policy_drift = (new_logp - old_logp).abs().mean()
        value_drift = (values - old_values).abs().mean()

        lstm_unit_metrics =\
            self.compute_lstm_unit_diagnostics(eval_output,
                                               mask_tb)

        loss = (
            policy_loss
            + self.state.cfg.ppo.vf_coef * value_loss
            - self.state.entropy_coef * entropy
        )

        if self.state.cfg.trainer.debug_mode is False:

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
                             grad_norm=grad_norm,
                             policy_drift=policy_drift,
                             value_drift=value_drift,
                             lstm_unit_metrics=lstm_unit_metrics)
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

    def compute_lstm_unit_diagnostics(
        self,
        eval_output: PolicyEvalOutput,
        mask: Optional[torch.Tensor]
        ) -> LSTMUnitDiagnostics:
        """
        Computes per-unit LSTM diagnostics (shape [H]) instead of scalars.
        """

        # (T, B, H)
        i_g = eval_output.gates.i_gates
        f_g = eval_output.gates.f_gates
        g_g = eval_output.gates.g_gates
        o_g = eval_output.gates.o_gates
        h_all = eval_output.new_hxs
        c_all = eval_output.new_cxs

        T, B, H = i_g.shape

        # ----------------------------------------------------
        # Normalize mask shape to (T, B)
        # ----------------------------------------------------
        if mask is None:
            mask_tb = None
        else:
            if mask.dim() == 1:
                # (T,) or (B,)
                if mask.shape[0] == T:
                    mask_tb = mask[:, None].expand(T, B)
                elif mask.shape[0] == B:
                    mask_tb = mask[None, :].expand(T, B)
                else:
                    raise ValueError("Mask length does not match T or B")
            elif mask.dim() == 2:
                if mask.shape == (B, T):
                    mask_tb = mask.transpose(0, 1)
                elif mask.shape == (T, B):
                    mask_tb = mask
                else:
                    raise ValueError("Mask must be (T,B) or (B,T)")
            else:
                raise ValueError("Mask must be 1D or 2D")

        # ----------------------------------------------------
        # Masked mean helper
        # ----------------------------------------------------
        def masked_mean(x):
            # x: (T, B, H)
            assert x.dim() == 3,\
                f"masked_mean expects (T,B,H), got {x.shape}"

            if mask_tb is None:
                return x.mean(dim=(0, 1)).detach()

            assert (mask_tb.shape[0] == x.shape[0] and
                    mask_tb.shape[1] == x.shape[1]), \
                f"mask/x mismatch: mask {mask_tb.shape}, x {x.shape}"

            m = mask_tb.unsqueeze(-1)  # (T, B, 1)
            x = x * m
            denom = m.sum(dim=(0, 1)).clamp(min=1)
            return (x.sum(dim=(0, 1)) / denom).detach()

        # ----------------------------------------------------
        # Per-unit means (shape [H])
        # ----------------------------------------------------
        i_mean = masked_mean(i_g)
        f_mean = masked_mean(f_g)
        g_mean = masked_mean(g_g)
        o_mean = masked_mean(o_g)
        
        # --- per-unit norms  ---
        h_norm = masked_mean(h_all)  # (H,)
        c_norm = masked_mean(c_all)  # (H,)

        # ----------------------------------------------------
        # Per-unit drift
        # ----------------------------------------------------
        prev = getattr(self.state,
                       "prev_lstm_unit_metrics",
                       None)

        if prev is None:
            i_drift = torch.zeros_like(i_mean)
            f_drift = torch.zeros_like(f_mean)
            g_drift = torch.zeros_like(g_mean)
            o_drift = torch.zeros_like(o_mean)
            h_drift = torch.zeros_like(h_norm)
            c_drift = torch.zeros_like(c_norm)
        else:
            i_drift = (i_mean - prev.i_mean).detach()
            f_drift = (f_mean - prev.f_mean).detach()
            g_drift = (g_mean - prev.g_mean).detach()
            o_drift = (o_mean - prev.o_mean).detach()
            h_drift = (h_norm - prev.h_norm).detach()
            c_drift = (c_norm - prev.c_norm).detach()

        # ----------------------------------------------------
        # Store for next iteration
        # ----------------------------------------------------
        self.state.prev_lstm_unit_metrics = SimpleNamespace(
            i_mean=i_mean,
            f_mean=f_mean,
            g_mean=g_mean,
            o_mean=o_mean,
            h_norm=h_norm,
            c_norm=c_norm
        )

        # ----------------------------------------------------
        # Saturation + entropy (vectorized)
        # ----------------------------------------------------
        sat = self.compute_gate_saturation_vectorized(eval_output,
                                                      mask_tb)

        ent = self.compute_gate_entropy_vectorized(eval_output.gates,
                                                   mask_tb)

        # ----------------------------------------------------
        # Return full per-unit metrics
        # ----------------------------------------------------
        return LSTMUnitDiagnostics(
            i_mean=i_mean,
            f_mean=f_mean,
            g_mean=g_mean,
            o_mean=o_mean,
            i_drift=i_drift,
            f_drift=f_drift,
            g_drift=g_drift,
            o_drift=o_drift,
            saturation=sat,
            entropy=ent,
            h_norm=h_norm,
            c_norm=c_norm,
            h_drift=h_drift,
            c_drift=c_drift,
            hidden_size=H
        )

    def compute_gate_saturation_vectorized(
        self,
        eval_output: PolicyEvalOutput,
        mask: Optional[torch.Tensor]
        ) -> LSTMGateSaturation:
        """
        Computes per-unit saturation metrics for all LSTM gates.
        Returns LSTMGateSaturation dataclass
        """

        # -------------------------
        # Extract gates (T, B, H)
        # -------------------------
        i_g = eval_output.gates.i_gates
        f_g = eval_output.gates.f_gates
        g_g = eval_output.gates.g_gates
        o_g = eval_output.gates.o_gates
        c_g = eval_output.gates.c_gates
        h_g = eval_output.gates.h_gates

        T, B, H = i_g.shape

        # -------------------------
        # Normalize mask to (T, B)
        # -------------------------
        if mask is None:
            mask_tb = None
        else:
            if mask.dim() == 1:
                if mask.shape[0] == T:
                    mask_tb = mask[:, None].expand(T, B)
                elif mask.shape[0] == B:
                    mask_tb = mask[None, :].expand(T, B)
                else:
                    raise ValueError("Mask length does not match T or B")
            elif mask.dim() == 2:
                if mask.shape == (B, T):
                    mask_tb = mask.transpose(0, 1)
                elif mask.shape == (T, B):
                    mask_tb = mask
                else:
                    raise ValueError("Mask must be (T,B) or (B,T)")
            else:
                raise ValueError("Mask must be 1D or 2D")

        # -------------------------
        # Helper: masked fraction
        # -------------------------
        def masked_fraction(x_bool):
            # x_bool: (T, B, H) boolean
            if mask_tb is None:
                return x_bool.float().mean(dim=(0, 1)).detach()

            m = mask_tb.unsqueeze(-1)  # (T, B, 1)
            x = x_bool.float() * m
            denom = m.sum(dim=(0, 1)).clamp(min=1)
            return (x.sum(dim=(0, 1)) / denom).detach()

        # -------------------------
        # Sigmoid gates: low/high saturation
        # -------------------------
        eps = self.state.cfg.trainer.gate_sat_eps

        i_sat_low  = masked_fraction(i_g < eps)
        i_sat_high = masked_fraction(i_g > 1 - eps)

        f_sat_low  = masked_fraction(f_g < eps)
        f_sat_high = masked_fraction(f_g > 1 - eps)

        o_sat_low  = masked_fraction(o_g < eps)
        o_sat_high = masked_fraction(o_g > 1 - eps)

        # -------------------------
        # Tanh-like gates: |x| > 1 - eps
        # -------------------------
        g_sat = masked_fraction(g_g.abs() > 1 - eps)
        c_sat = masked_fraction(c_g.abs() > 1 - eps)
        h_sat = masked_fraction(h_g.abs() > 1 - eps)

        # -------------------------
        # Return dictionary for LSTMUnitDiagnostics
        # -------------------------
        return LSTMGateSaturation(
            i_sat_low=i_sat_low,
            i_sat_high=i_sat_high,
            f_sat_low=f_sat_low,
            f_sat_high=f_sat_high,
            o_sat_low=o_sat_low,
            o_sat_high=o_sat_high,
            g_sat=g_sat,
            c_sat=c_sat,
            h_sat=h_sat,
            hidden_size=H
        )

    def compute_gate_entropy_vectorized(
        self,
        gates: LSTMGates,
        mask: Optional[torch.Tensor]
        ) -> LSTMGateEntropy:
        """
        Computes per-unit entropy for all LSTM gates.
        Returns LSTMGateEntropy dataclass with 1-D tensors [H].
        """

        # (T, B, H)
        i_g = gates.i_gates
        f_g = gates.f_gates
        g_g = gates.g_gates
        o_g = gates.o_gates

        # cell/hidden states (T, B, H)
        c_g = gates.c_gates if hasattr(gates, "c_gates") else None
        h_g = gates.h_gates if hasattr(gates, "h_gates") else None

        T, B, H = i_g.shape

        # ----------------------------------------------------
        # Normalize mask to (T, B)
        # ----------------------------------------------------
        if mask is None:
            mask_tb = None
        else:
            if mask.dim() == 1:
                if mask.shape[0] == T:
                    mask_tb = mask[:, None].expand(T, B)
                elif mask.shape[0] == B:
                    mask_tb = mask[None, :].expand(T, B)
                else:
                    raise ValueError("Mask length mismatch")
            elif mask.dim() == 2:
                if mask.shape == (B, T):
                    mask_tb = mask.transpose(0, 1)
                elif mask.shape == (T, B):
                    mask_tb = mask
                else:
                    raise ValueError("Mask must be (T,B) or (B,T)")
            else:
                raise ValueError("Mask must be 1D or 2D")

        # ----------------------------------------------------
        # Helper: masked mean over T,B
        # ----------------------------------------------------
        def masked_mean(x):
            if mask_tb is None:
                return x.mean(dim=(0, 1)).detach()

            m = mask_tb.unsqueeze(-1)  # (T, B, 1)
            x = x * m
            denom = m.sum(dim=(0, 1)).clamp(min=1)
            return (x.sum(dim=(0, 1)) / denom).detach()

        # ----------------------------------------------------
        # Sigmoid gate entropy: H(g) = -g log g - (1-g) log (1-g)
        # ----------------------------------------------------
        eps = self.state.cfg.trainer.gate_ent_eps

        def sigmoid_entropy(g):
            g = g.clamp(eps, 1 - eps)
            return -(g * torch.log(g) + (1 - g) * torch.log(1 - g))

        i_entropy = masked_mean(sigmoid_entropy(i_g))
        f_entropy = masked_mean(sigmoid_entropy(f_g))
        o_entropy = masked_mean(sigmoid_entropy(o_g))

        # ----------------------------------------------------
        # Tanh gate entropy: convert x ∈ [-1,1] → p ∈ [0,1]
        # ----------------------------------------------------
        def tanh_entropy(x):
            p = ((x + 1) * 0.5).clamp(eps, 1 - eps)
            return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

        g_entropy = masked_mean(tanh_entropy(g_g))

        c_entropy = masked_mean(tanh_entropy(c_g)) if c_g is not None else torch.zeros(H)
        h_entropy = masked_mean(tanh_entropy(h_g)) if h_g is not None else torch.zeros(H)

        return LSTMGateEntropy(
            i_entropy=i_entropy,
            f_entropy=f_entropy,
            o_entropy=o_entropy,
            g_entropy=g_entropy,
            c_entropy=c_entropy,
            h_entropy=h_entropy,
            hidden_size=H
        )

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

    def save_checkpoint(self):

        checkpoint_pth =\
            self.checkpoint_dir.joinpath(self.state.cfg.log.run_name +
                                         "_checkpoint_" +
                                         f"{self.state.update_idx}.pt")
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_idx": self.state.update_idx,
            "trainer_state": {
                "update_idx": self.state.update_idx,
                "lr": float(self.state.lr),
                "entropy_coef": float(self.state.entropy_coef),
                "clip_range": float(self.state.clip_range),
                "target_kl": float(self.state.target_kl),
                "early_stopping_kl": float(self.state.early_stopping_kl),
            }
        }, checkpoint_pth)

    def load_checkpoint(self,
                        checkpoint):

        trainer_state = checkpoint["trainer_state"]

        self.state.update_idx = trainer_state["update_idx"]
        self.state.lr = trainer_state["lr"]
        self.state.entropy_coef = trainer_state["entropy_coef"]
        self.state.clip_range = trainer_state["clip_range"]
        self.state.target_kl = trainer_state["target_kl"]
        self.state.early_stopping_kl = trainer_state["early_stopping_kl"]

    def validate_tbptt(self, K=16):

        self.state.reset(1)
        self.state.init_stats()

        self.policy.eval()
        self.collect_rollout()

        batch = next(self.buffer.get_recurrent_minibatches())

        print("batch.hxs shape:", batch.hxs.shape)
        print("batch.cxs shape:", batch.cxs.shape)

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

                hxs = out.new_hxs[-1].detach()   # (B, H)
                cxs = out.new_cxs[-1].detach()   # (B, H)

        vals_rec = torch.cat(vals_chunks, dim=0)
        logp_rec = torch.cat(logp_chunks, dim=0)

        max_val_diff = (vals_rec - full_vals).abs().max().item()
        max_logp_diff = (logp_rec - full_logp).abs().max().item()

        print("TBPTT value diff:", max_val_diff)
        print("TBPTT logprob diff:", max_logp_diff)

        assert max_val_diff < 1e-6
        assert max_logp_diff < 1e-6

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

            policy_in = env_state.policy_input
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

            policy_in = env_state.policy_input
            hxs_trace.append(policy_in.hxs.clone())
            cxs_trace.append(policy_in.cxs.clone())

            actions, _, _ = self.policy.act(policy_in)
            env_state = self.env.step(actions)

        return hxs_trace, cxs_trace
    
    def validate_lstm_state_flow(self):

        print("=== LSTM State-Flow Validation ===")

        # -----------------------------------------------------
        # 1. Deterministic mode: no DropConnect, no dropout
        # -----------------------------------------------------
        self.policy.eval()
        self.state.cfg.trainer.debug_mode = True

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # -----------------------------------------------------
        # 2. Collect rollout using *forward()*, not act()
        # -----------------------------------------------------
        self.buffer.reset()
        env_state = self.env.reset(seed=self.state.cfg.trainer.seed)

        # Use stored LSTM states if available
        self.env.set_initial_lstm_states(self.buffer.get_last_lstm_states())

        stored_values = []
        stored_logprobs = []
        stored_hxs = []
        stored_cxs = []
        stored_obs = []
        stored_actions = []

        with torch.no_grad():
            for _ in range(self.rollout_steps):

                policy_in = env_state.policy_input

                # IMPORTANT: use forward(), not act()
                policy_out = self.policy.forward(policy_in)

                dist = Categorical(logits=policy_out.logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

                stored_obs.append(env_state.obs.clone())
                stored_values.append(policy_out.values.clone())
                stored_logprobs.append(logprobs.clone())
                
                # PRE-STEP hidden state: what was actually used
                stored_hxs.append(policy_in.hxs.clone())
                stored_cxs.append(policy_in.cxs.clone())

                stored_actions.append(actions.clone())
                env_state = self.env.step(actions)

        # Stack rollout tensors
        stored_obs = torch.stack(stored_obs, dim=0)          # (T, B, obs_dim)
        stored_values = torch.stack(stored_values, dim=0)    # (T, B)
        stored_logprobs = torch.stack(stored_logprobs, dim=0)# (T, B)
        stored_hxs = torch.stack(stored_hxs, dim=0)          # (T, B, H)
        stored_cxs = torch.stack(stored_cxs, dim=0)          # (T, B, H)
        stored_actions = torch.stack(stored_actions, dim=0)  # (T, B)

        T, B, _ = stored_obs.shape
        print(f"T = {T}, B = {B}")

        # -----------------------------------------------------
        # 3. Replay using EXACT SAME forward path
        # -----------------------------------------------------
        rec_values = []
        rec_logprobs = []

        with torch.no_grad():
            for t in range(T):

                policy_in = PolicyInput(
                    obs=stored_obs[t],      # (B, obs_dim)
                    hxs=stored_hxs[t],      # (B, H)
                    cxs=stored_cxs[t],      # (B, H)
                )

                policy_out = self.policy.forward(policy_in)
                dist = Categorical(logits=policy_out.logits)

                actions = stored_actions[t]
                logp_t = dist.log_prob(actions)

                rec_values.append(policy_out.values.unsqueeze(0))
                rec_logprobs.append(logp_t.unsqueeze(0))

        values_rec = torch.cat(rec_values, dim=0)
        logprobs_rec = torch.cat(rec_logprobs, dim=0)

        # -----------------------------------------------------
        # 4. Compare
        # -----------------------------------------------------
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

# https://realpython.com/python-mixin/
from .trainer_renderers import (
    render_ppo_table,
    render_episode_table,
    render_episode_trends,
    render_policy_stability,
    render_value_drift,
    render_histogram,
    render_env_timelines,
)

# Attach as methods
LSTMPPOTrainer.render_ppo_table = render_ppo_table
LSTMPPOTrainer.render_episode_table = render_episode_table
LSTMPPOTrainer.render_episode_trends = render_episode_trends
LSTMPPOTrainer.render_policy_stability = render_policy_stability
LSTMPPOTrainer.render_value_drift = render_value_drift
LSTMPPOTrainer.render_histogram = render_histogram
LSTMPPOTrainer.render_env_timelines = render_env_timelines


def train(total_updates=2000):

    cfg = Config()

    cfg = initialize_config(cfg)

    trainer = LSTMPPOTrainer(cfg)

    trainer.train(total_updates=total_updates)


def validate():

    trainer = LSTMPPOTrainer.for_validation()

    trainer.assert_hidden_state_flow()
    trainer.assert_rollout_deterministic()
    trainer.validate_tbptt()
    trainer.validate_lstm_state_flow()
