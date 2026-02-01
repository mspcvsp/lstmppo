import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy, ZeroFeatureEncoder
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.policy


def test_policy_lstm_only_mode_unroll(trainer_state: TrainerState):
    """
    Smoke test: when flat_obs_dim == 0 and action_dim > 0,
    the policy must:
      - use ZeroFeatureEncoder for the encoder
      - run the LSTM purely on its recurrent state
      - support multi-step unroll without shape errors
      - produce stable, correctly shaped outputs at each step
    """
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 0  # ZeroFeatureEncoder path
    trainer_state.env_info.action_dim = 3  # actor active

    trainer_state.cfg.lstm.enc_hidden_size = 16
    trainer_state.cfg.lstm.lstm_hidden_size = 8

    policy = LSTMPPOPolicy(trainer_state)

    # Encoder must be ZeroFeatureEncoder
    assert isinstance(policy.encoder, ZeroFeatureEncoder)

    B = 4
    T = 5
    H = trainer_state.cfg.lstm.lstm_hidden_size

    # Zero observations for all timesteps
    obs = torch.zeros(B, T, 0)

    # Initial hidden state
    hxs = torch.zeros(B, H)
    cxs = torch.zeros(B, H)

    # Unroll manually across time
    for t in range(T):
        out = policy.forward(
            PolicyInput(
                obs=obs[:, t],  # (B, 0)
                hxs=hxs,
                cxs=cxs,
            )
        )

        # Check shapes at each step
        assert out.logits.shape == (B, trainer_state.env_info.action_dim)
        assert out.values.shape == (B,)
        assert out.new_hxs.shape == (B, H)
        assert out.new_cxs.shape == (B, H)

        # Feed hidden state forward
        hxs, cxs = out.new_hxs, out.new_cxs
