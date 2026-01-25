import torch
import torch.nn as nn
from lstmppo.policy import LSTMPPOPolicy, ZeroFeatureEncoder
from lstmppo.types import Config, PolicyInput


def test_policy_encoder_and_actor_identity_paths():
    """
    Smoke test: when both flat_obs_dim == 0 and action_dim == 0,
    the policy must:
      - use ZeroFeatureEncoder for the encoder
      - use nn.Identity() for the actor head
      - avoid constructing any Linear(0, H) or Linear(H, 0)
      - produce correctly shaped outputs for a dummy forward pass
    """

    cfg = Config()

    # Force both identity paths
    cfg.env.obs_space = None
    cfg.env.flat_obs_dim = 0
    cfg.env.action_dim = 0

    cfg.lstm.enc_hidden_size = 16
    cfg.lstm.lstm_hidden_size = 8

    policy = LSTMPPOPolicy(cfg)

    # --- Encoder must be ZeroFeatureEncoder ---
    assert isinstance(policy.encoder, ZeroFeatureEncoder)

    # --- Actor must be Identity ---
    assert isinstance(policy.actor, nn.Identity)

    # --- Forward pass must work with both identity paths ---
    B = 4
    H = cfg.lstm.lstm_hidden_size

    obs = torch.zeros(B, 0)
    hxs = torch.zeros(B, H)
    cxs = torch.zeros(B, H)

    out = policy.forward(
        PolicyInput(obs=obs, hxs=hxs, cxs=cxs)
    )

    # --- Output sanity checks ---
    assert out.logits.shape == (B, 0)
    assert out.values.shape == (B,)
    assert out.new_hxs.shape == (B, H)
    assert out.new_cxs.shape == (B, H)