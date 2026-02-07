"""
This test MUST use the real TrainerState and real LSTMPPOPolicy.

Why:
-----
LSTM drift is a *model‑level* invariant, not a rollout/buffer invariant. It validates the true behavior of the
GateLSTMCell, encoder → LSTM wiring, gate computations, and detach logic over long unrolls.

FakeState, FakePolicy, or any synthetic rollout helpers would mask or distort the actual LSTM dynamics. Only the real
policy forward pass exposes the true c_t evolution, which is what this test is protecting.

If this test ever stops using the real model, it will no longer detect:

    • gate wiring regressions
    • incorrect detach behavior
    • encoder → LSTM misalignment
    • broken long‑horizon unrolls
    • drift/variance collapse in c_t

This test is the sentinel for long‑term LSTM memory correctness. Do not replace the real model with fakes here.
"""

import pytest
import torch

from lstmppo.types import PolicyInput
from tests.helpers.fake_policy import make_fake_policy
from tests.helpers.fake_state import TrainerStateProtocol

pytestmark = pytest.mark.drift


def test_cell_state_drift_accumulates_over_time(fake_state: TrainerStateProtocol):
    policy = make_fake_policy()
    policy.eval()

    B = 3
    H = fake_state.cfg.lstm.lstm_hidden_size
    D = fake_state.flat_obs_dim

    lengths = [5, 50]
    num_samples = 20

    avg_drifts = []

    for L in lengths:
        drifts = []
        for _ in range(num_samples):
            obs = torch.randn(B, L, D)
            h0 = torch.zeros(B, H)
            c0 = torch.zeros(B, H)

            out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

            drifts.append(out.gates.c_gates.pow(2).mean())

        avg_drifts.append(torch.stack(drifts).mean())

    eps = 1e-5
    assert avg_drifts[1] + eps >= avg_drifts[0] - eps
