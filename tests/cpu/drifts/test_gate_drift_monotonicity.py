"""
Verifies gate drift increases when hidden state magnitude increases
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_gate_drift_monotonicity(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)

    B, T = 3, 5
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    h0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)

    # Baseline
    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    drift1 = out1.gates.h_gates.pow(2).mean()

    # Synthetic drift: scale obs
    out2 = policy.forward(PolicyInput(obs=obs * 3.0, hxs=h0, cxs=c0))
    drift2 = out2.gates.h_gates.pow(2).mean()

    assert drift2 > drift1
