# lstmppo/logging/tensorboard_logger.py

from __future__ import annotations

import torch
from torch.utils.tensorboard import SummaryWriter

from lstmppo.types import (
    LSTMGates,
    LSTMUnitDiagnostics,
    Metrics,
)


class TensorboardLogger:
    def __init__(self, logdir: str, run_name: str):
        self.logdir = logdir
        self.run_name = run_name
        self.writer = SummaryWriter(log_dir=f"{logdir}/{run_name}")

    # ------------------------------------------------------------------
    # PPO / high-level scalars
    # ------------------------------------------------------------------
    def log_ppo_scalars(self, step: int, metrics: Metrics, lr: float, entropy_coef: float, clip_range: float):
        w = self.writer

        w.add_scalar("ppo/return/avg", metrics.avg_ep_returns, step)
        w.add_scalar("ppo/return/max", metrics.max_ep_returns, step)
        w.add_scalar("ppo/length/avg", metrics.avg_ep_len, step)
        w.add_scalar("ppo/length/max", metrics.max_ep_len, step)

        w.add_scalar("ppo/loss/policy", metrics.policy_loss, step)
        w.add_scalar("ppo/loss/value", metrics.value_loss, step)
        w.add_scalar("ppo/entropy", metrics.entropy, step)
        w.add_scalar("ppo/kl", metrics.approx_kl, step)
        w.add_scalar("ppo/clip_frac", metrics.clip_frac, step)
        w.add_scalar("ppo/grad_norm", metrics.grad_norm, step)
        w.add_scalar("ppo/explained_var", metrics.explained_var, step)

        w.add_scalar("ppo/hparams/lr", lr, step)
        w.add_scalar("ppo/hparams/entropy_coef", entropy_coef, step)
        w.add_scalar("ppo/hparams/clip_range", clip_range, step)

        w.add_scalar("ppo/drift/policy_drift", metrics.policy_drift, step)
        w.add_scalar("ppo/drift/value_drift", metrics.value_drift, step)

    # ------------------------------------------------------------------
    # LSTM scalar summaries (per update)
    # ------------------------------------------------------------------
    def log_lstm_scalars(self, step: int, diag: LSTMUnitDiagnostics):
        w = self.writer

        if diag.h_norm is not None:
            w.add_scalar("lstm/norms/h_norm", diag.h_norm.mean().item(), step)
        if diag.c_norm is not None:
            w.add_scalar("lstm/norms/c_norm", diag.c_norm.mean().item(), step)

        if diag.h_drift is not None:
            w.add_scalar("lstm/drift/h_drift", diag.h_drift.mean().item(), step)
        if diag.c_drift is not None:
            w.add_scalar("lstm/drift/c_drift", diag.c_drift.mean().item(), step)

        if diag.i_mean is not None:
            w.add_scalar("lstm/gates/i_mean", diag.i_mean.mean().item(), step)
        if diag.f_mean is not None:
            w.add_scalar("lstm/gates/f_mean", diag.f_mean.mean().item(), step)
        if diag.g_mean is not None:
            w.add_scalar("lstm/gates/g_mean", diag.g_mean.mean().item(), step)
        if diag.o_mean is not None:
            w.add_scalar("lstm/gates/o_mean", diag.o_mean.mean().item(), step)

        if diag.entropy is not None:
            ent = diag.entropy
            w.add_scalar("lstm/entropy/i_entropy", ent.i_entropy.mean().item(), step)
            w.add_scalar("lstm/entropy/f_entropy", ent.f_entropy.mean().item(), step)
            w.add_scalar("lstm/entropy/o_entropy", ent.o_entropy.mean().item(), step)
            w.add_scalar("lstm/entropy/g_entropy", ent.g_entropy.mean().item(), step)
            w.add_scalar("lstm/entropy/c_entropy", ent.c_entropy.mean().item(), step)
            w.add_scalar("lstm/entropy/h_entropy", ent.h_entropy.mean().item(), step)

        if diag.saturation is not None:
            sat = diag.saturation
            w.add_scalar("lstm/saturation/i_sat_low", sat.i_sat_low.mean().item(), step)
            w.add_scalar("lstm/saturation/i_sat_high", sat.i_sat_high.mean().item(), step)
            w.add_scalar("lstm/saturation/f_sat_low", sat.f_sat_low.mean().item(), step)
            w.add_scalar("lstm/saturation/f_sat_high", sat.f_sat_high.mean().item(), step)
            w.add_scalar("lstm/saturation/o_sat_low", sat.o_sat_low.mean().item(), step)
            w.add_scalar("lstm/saturation/o_sat_high", sat.o_sat_high.mean().item(), step)
            w.add_scalar("lstm/saturation/g_sat", sat.g_sat.mean().item(), step)
            w.add_scalar("lstm/saturation/c_sat", sat.c_sat.mean().item(), step)
            w.add_scalar("lstm/saturation/h_sat", sat.h_sat.mean().item(), step)

    # ------------------------------------------------------------------
    # LSTM per-unit histograms (every N updates)
    # ------------------------------------------------------------------
    def log_lstm_histograms(self, step: int, diag: LSTMUnitDiagnostics):
        w = self.writer

        if diag.i_mean is not None:
            w.add_histogram("lstm/gates/i_mean_units", diag.i_mean, step)
        if diag.f_mean is not None:
            w.add_histogram("lstm/gates/f_mean_units", diag.f_mean, step)
        if diag.g_mean is not None:
            w.add_histogram("lstm/gates/g_mean_units", diag.g_mean, step)
        if diag.o_mean is not None:
            w.add_histogram("lstm/gates/o_mean_units", diag.o_mean, step)

        if diag.h_norm is not None:
            w.add_histogram("lstm/norms/h_norm_units", diag.h_norm, step)
        if diag.c_norm is not None:
            w.add_histogram("lstm/norms/c_norm_units", diag.c_norm, step)

        if diag.h_drift is not None:
            w.add_histogram("lstm/drift/h_drift_units", diag.h_drift, step)
        if diag.c_drift is not None:
            w.add_histogram("lstm/drift/c_drift_units", diag.c_drift, step)

        if diag.saturation is not None:
            sat = diag.saturation
            w.add_histogram("lstm/saturation/i_sat_low_units", sat.i_sat_low, step)
            w.add_histogram("lstm/saturation/i_sat_high_units", sat.i_sat_high, step)
            w.add_histogram("lstm/saturation/f_sat_low_units", sat.f_sat_low, step)
            w.add_histogram("lstm/saturation/f_sat_high_units", sat.f_sat_high, step)
            w.add_histogram("lstm/saturation/o_sat_low_units", sat.o_sat_low, step)
            w.add_histogram("lstm/saturation/o_sat_high_units", sat.o_sat_high, step)

        if diag.entropy is not None:
            ent = diag.entropy
            w.add_histogram("lstm/entropy/i_entropy_units", ent.i_entropy, step)
            w.add_histogram("lstm/entropy/f_entropy_units", ent.f_entropy, step)
            w.add_histogram("lstm/entropy/o_entropy_units", ent.o_entropy, step)
            w.add_histogram("lstm/entropy/g_entropy_units", ent.g_entropy, step)
            w.add_histogram("lstm/entropy/c_entropy_units", ent.c_entropy, step)
            w.add_histogram("lstm/entropy/h_entropy_units", ent.h_entropy, step)

    # ------------------------------------------------------------------
    # LSTM heatmaps (T, H) from gates (every N rollouts)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_for_image(x: torch.Tensor) -> torch.Tensor:
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x

    def log_lstm_heatmaps(self, step: int, gates: LSTMGates):
        w = self.writer

        # gates.*: (T, B, H) → pick B=1 and reshape to (1, T, H)
        def pick_b0(t: torch.Tensor) -> torch.Tensor:
            # (T, B, H) -> (1, T, H)
            return t[:, 0].unsqueeze(0)

        i_map = pick_b0(gates.i_gates)
        f_map = pick_b0(gates.f_gates)
        g_map = pick_b0(gates.g_gates)
        o_map = pick_b0(gates.o_gates)
        c_map = pick_b0(gates.c_gates)
        h_map = pick_b0(gates.h_gates)

        w.add_image("lstm/heatmap/i_gate", self._normalize_for_image(i_map), step, dataformats="CHW")
        w.add_image("lstm/heatmap/f_gate", self._normalize_for_image(f_map), step, dataformats="CHW")
        w.add_image("lstm/heatmap/g_gate", self._normalize_for_image(g_map), step, dataformats="CHW")
        w.add_image("lstm/heatmap/o_gate", self._normalize_for_image(o_map), step, dataformats="CHW")
        w.add_image("lstm/heatmap/c_gate", self._normalize_for_image(c_map), step, dataformats="CHW")
        w.add_image("lstm/heatmap/h_gate", self._normalize_for_image(h_map), step, dataformats="CHW")

    def close(self):
        self.writer.close()
