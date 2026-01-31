import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_tanh_gate_saturation_increases_with_extreme_inputs(trainer_state: TrainerState):
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

    g1 = out1.gates.g_gates
    g2 = out2.gates.g_gates

    # Tanh saturation metric: 1 - |g| decreases as saturation increases
    sat1 = (1 - g1.abs()).mean()
    sat2 = (1 - g2.abs()).mean()

    assert sat2 < sat1
