"""
Ensures drift grows smoothly with sequence length — no sudden jumps or
collapses. This test ensures drift doesn’t explode or collapse between
adjacent horizons.
"""

import pytest
import torch
from lstmppo.diagnostics.recurrent import compute_drift_sequence

from lstmppo.policy import LSTMPPOPolicy


@pytest.mark.drift
def test_drift_growth_smoothness():
    """
    Drift should grow smoothly on average across multiple sequences.
    Individual sequences may fluctuate due to float32 noise, so we test
    the averaged drift curve and its smoothness properties.
    """

    torch.manual_seed(0)

    # Number of independent drift rollouts to average over
    num_sequences = 32
    T = 64  # sequence length

    policy = LSTMPPOPolicy(
        obs_dim=8,
        action_dim=4,
        hidden_size=32,
    )

    drifts = []

    for _ in range(num_sequences):
        drift = compute_drift_sequence(policy, T=T)
        assert drift.shape == (T,)
        drifts.append(drift)

    # (N, T)
    drift_tensor = torch.stack(drifts, dim=0)

    # Mean drift over sequences
    mean_drift = drift_tensor.mean(dim=0)

    # 1. Mean drift should be non-decreasing within tolerance
    diff = mean_drift[1:] - mean_drift[:-1]
    assert torch.all(diff >= -1e-6), "Mean drift should not decrease beyond tiny tolerance"

    # 2. Smoothness: no large jumps
    assert diff.abs().max() < 1e-3, "Mean drift should change smoothly without large jumps"

    # 3. Variance should remain bounded
    var = drift_tensor.var(dim=0)
    assert var.max() < 1e-3, "Drift variance across sequences should remain stable"
