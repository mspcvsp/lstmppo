"""
This test catches:

- accidental shape changes
- wrong flattening
- wrong output dimension
"""

import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput


def test_policy_forward_shapes(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    model = LSTMPPOPolicy(trainer_state)

    B = 3

    inp = PolicyInput(
        obs=torch.randn(B, trainer_state.env_info.flat_obs_dim),
        hxs=torch.randn(B, trainer_state.cfg.lstm.lstm_hidden_size),
        cxs=torch.randn(B, trainer_state.cfg.lstm.lstm_hidden_size),
    )

    outp = model(inp)

    assert outp.logits.shape == (B, trainer_state.env_info.action_dim)
    assert outp.values.shape == (B,)
    assert outp.new_hxs.shape == (B, trainer_state.cfg.lstm.lstm_hidden_size)
    assert outp.new_cxs.shape == (B, trainer_state.cfg.lstm.lstm_hidden_size)
    assert outp.ar_loss.shape == torch.Size([])
    assert outp.tar_loss.shape == torch.Size([])

    for name in ["i_gates", "f_gates", "o_gates", "o_gates", "c_gates", "h_gates"]:
        t = getattr(outp.gates, name)
        assert t.shape == (B, 1, trainer_state.cfg.lstm.lstm_hidden_size)
