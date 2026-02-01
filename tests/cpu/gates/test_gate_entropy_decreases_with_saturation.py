"""
Ensures entropy decreases when gates saturate.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_gate_entropy_decreases_with_saturation(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 5
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    h0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)

    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))

    i1 = out1.gates.i_gates
    i2 = out2.gates.i_gates

    eps = 1e-8

    ent1 = -(i1 * torch.log(i1 + eps) + (1 - i1) * torch.log(1 - i1 + eps)).mean()

    ent2 = -(i2 * torch.log(i2 + eps) + (1 - i2) * torch.log(1 - i2 + eps)).mean()

    assert ent2 < ent1
