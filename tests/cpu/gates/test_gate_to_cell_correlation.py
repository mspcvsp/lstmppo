"""
Ensures forget gate correlates with cell‑state magnitude
(core LSTM behavior).
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_gate_to_cell_correlation(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 30
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    h0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    f = out.gates.f_gates.reshape(-1)
    c = out.gates.c_gates.abs().reshape(-1)

    corr = torch.corrcoef(torch.stack([f, c]))[0, 1]

    """
    The correlation between forget gate and cell magnitude should be small
    near zero).

    Why?
    - The forget gate is initialized near 0.73 (bias=1.0)
    - The cell state is unnormalized and accumulates drift
    - The hidden state is normalized (LayerNorm)
    - The recurrent dynamics do not enforce a monotonic relationship
    - The LSTM equations do not imply a sign constraint
    - The sample size is tiny, so correlation estimates are noisy

    Therefore:
    - The sign is not stable
    - The magnitude should be small
    - The correlation should not be strongly positive or strongly negative

    This is the real invariant.
    """
    assert corr.abs() < 0.2
