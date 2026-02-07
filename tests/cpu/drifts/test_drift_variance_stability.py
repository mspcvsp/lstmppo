"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Drift variance stability is a *model‑level* invariant of the GateLSTMCell. It validates that hidden‑state drift does
not oscillate wildly over time, which would indicate:

    • incorrect gate wiring
    • unstable long‑horizon LSTM unrolls
    • encoder → LSTM integration errors
    • detach or state‑update bugs
    • pathological gate activations

Any fake state, fake policy, or synthetic rollout would bypass the true LSTM dynamics and invalidate the signal this
test is designed to catch. This is a sentinel test for long‑sequence LSTM stability. Do not replace the real model
here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_drift_variance_stability(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3
    trainer_state.cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 100
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    # Long unroll to expose drift variance behavior
    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    # Drift per timestep: squared hidden gate magnitude
    drift_t = out.gates.h_gates.pow(2).mean(dim=(0, 2))  # (T,)
    var = drift_t.var()

    # Variance should be finite and not extreme
    assert torch.isfinite(var)
    assert var < 0.01
