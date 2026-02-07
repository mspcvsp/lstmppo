"""
This test MUST use the real TrainerState and real LSTMPPOPolicy.

Why:
-----
Drift growth rate is a *model‑level* invariant of the GateLSTMCell.
It validates the true long‑horizon behavior of the LSTM:
    • gate wiring correctness
    • encoder → LSTM integration
    • hidden‑state update equations
    • detach logic
    • variance accumulation over time

Any fake state, fake policy, or synthetic rollout would bypass the
actual LSTM math and completely invalidate the signal this test is
designed to catch.

This test is a sentinel for long‑sequence LSTM stability. Do not
replace the real model here.
"""

import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import Config, PolicyInput


def test_drift_growth_rate():
    cfg = Config()
    cfg.trainer.debug_mode = True

    state = TrainerState(cfg)

    # Explicitly set env info for deterministic dimensions
    assert state.env_info is not None
    state.env_info.flat_obs_dim = 4
    state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(state)
    policy.eval()

    B = 3
    H = cfg.lstm.lstm_hidden_size
    D = state.env_info.flat_obs_dim

    lengths = [20, 40, 80]
    num_samples = 20  # more samples = smoother averages

    avg_drifts = []

    for L in lengths:
        drifts = []

        for _ in range(num_samples):
            obs = torch.randn(B, L, D)
            h0 = torch.zeros(B, H)
            c0 = torch.zeros(B, H)

            out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
            drifts.append(out.gates.h_gates.pow(2).mean())

        avg_drifts.append(torch.stack(drifts).mean())

    # Allow tiny decreases due to noise
    eps = 1e-5
    assert avg_drifts[1] + eps >= avg_drifts[0] - eps
    assert avg_drifts[2] + eps >= avg_drifts[1] - eps
