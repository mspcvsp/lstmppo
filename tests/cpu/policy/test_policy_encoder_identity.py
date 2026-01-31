import torch

from lstmppo.policy import LSTMPPOPolicy, ZeroFeatureEncoder
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput


def test_policy_encoder_identity_path(trainer_state: TrainerState):
    """
    Smoke test: when obs_dim == 0, the policy must:
      - use nn.Identity() as the encoder
      - avoid constructing Linear(0, H)
      - produce correctly shaped outputs for a dummy forward pass
    """

    # Narrow Optional for Pylance
    assert trainer_state.env_info is not None

    # Force identity encoder path
    trainer_state.env_info.flat_obs_dim = 0
    trainer_state.env_info.action_dim = 2  # keep actor valid

    trainer_state.cfg.lstm.enc_hidden_size = 16
    trainer_state.cfg.lstm.lstm_hidden_size = 8

    policy = LSTMPPOPolicy(trainer_state)

    # --- Encoder must be Identity ---
    assert isinstance(policy.encoder, ZeroFeatureEncoder)

    # --- Forward pass must work with obs_dim == 0 ---
    B = 4
    H = trainer_state.cfg.lstm.lstm_hidden_size

    obs = torch.zeros(B, 0)  # shape matches obs_dim == 0
    hxs = torch.zeros(B, H)
    cxs = torch.zeros(B, H)

    out = policy.forward(inp=PolicyInput(obs=obs, hxs=hxs, cxs=cxs))

    # --- Output sanity checks ---
    assert out.logits.shape == (B, trainer_state.env_info.action_dim)
    assert out.values.shape == (B,)
    assert out.new_hxs.shape == (B, H)
    assert out.new_cxs.shape == (B, H)
