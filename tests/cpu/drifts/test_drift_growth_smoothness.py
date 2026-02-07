"""
This test MUST use the real LSTMPPOPolicy, real TrainerState, and the real
compute_drift_sequence() path.

Why:
-----
Drift smoothness is a *model‑level* invariant of the GateLSTMCell and the diagnostics pipeline. It validates:

    • correct gate wiring
    • stable long‑horizon LSTM unrolls
    • encoder → LSTM → drift‑capture integration
    • absence of drift explosions or collapses
    • statistical smoothness of drift over time

Any fake state, fake policy, or synthetic rollout would bypass the true LSTM dynamics and invalidate the signal this
test is designed to catch.

This is a sentinel test for long‑sequence LSTM stability and diagnostic correctness. Do not replace the real model
here.
"""

import pytest
import torch

from lstmppo.diagnostics.recurrent import compute_drift_sequence
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import Config


@pytest.mark.drift
def test_drift_growth_smoothness():
    torch.manual_seed(0)

    # --- Real trainer state ---
    cfg = Config()
    cfg.trainer.debug_mode = True
    cfg.lstm.lstm_hidden_size = 32

    state = TrainerState(cfg)
    assert state.env_info is not None
    state.env_info.flat_obs_dim = 8
    state.env_info.action_dim = 4

    # --- Real policy ---
    policy = LSTMPPOPolicy(state)
    policy.eval()

    # Drift parameters
    num_sequences = 32
    T = 64

    # Collect drift curves
    drifts = []
    for _ in range(num_sequences):
        drift = compute_drift_sequence(policy, T=T)
        assert drift.shape == (T,)
        drifts.append(drift)

    drift_tensor = torch.stack(drifts, dim=0)

    # Mean drift over sequences
    mean_drift = drift_tensor.mean(dim=0)

    # 1. Non-decreasing drift (within tolerance)
    diff = mean_drift[1:] - mean_drift[:-1]
    assert torch.all(diff >= -1e-6), "Mean drift should not decrease beyond tiny tolerance"

    # 2. Smoothness: no large jumps
    assert diff.abs().max() < 1e-3, "Mean drift should change smoothly without large jumps"

    # 3. Variance should remain bounded
    var = drift_tensor.var(dim=0)
    assert var.max() < 1e-3, "Drift variance across sequences should remain stable"
