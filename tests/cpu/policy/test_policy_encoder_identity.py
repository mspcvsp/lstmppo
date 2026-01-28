import torch

from lstmppo.policy import LSTMPPOPolicy, ZeroFeatureEncoder
from lstmppo.types import Config, PolicyInput


def test_policy_encoder_identity_path():
    """
    Smoke test: when obs_dim == 0, the policy must:
      - use nn.Identity() as the encoder
      - avoid constructing Linear(0, H)
      - produce correctly shaped outputs for a dummy forward pass
    """

    cfg = Config()

    # Force identity encoder path WITHOUT mutating read-only obs_dim
    cfg.env.obs_space = None
    cfg.env.flat_obs_dim = 0        # this drives cfg.obs_dim internally

    cfg.env.action_dim = 2          # keep actor valid
    cfg.lstm.enc_hidden_size = 16
    cfg.lstm.lstm_hidden_size = 8

    policy = LSTMPPOPolicy(cfg)

    # --- Encoder must be Identity ---
    assert isinstance(policy.encoder, ZeroFeatureEncoder)

    # --- Forward pass must work with obs_dim == 0 ---
    B = 4
    H = cfg.lstm.lstm_hidden_size

    obs = torch.zeros(B, 0)         # shape matches obs_dim == 0
    hxs = torch.zeros(B, H)
    cxs = torch.zeros(B, H)

    out = policy.forward(
        inp=PolicyInput(obs=obs,
                        hxs=hxs,
                        cxs=cxs)
    )

    # --- Output sanity checks ---
    assert out.logits.shape == (B, cfg.env.action_dim)
    assert out.values.shape == (B,)
    assert out.new_hxs.shape == (B, H)
    assert out.new_cxs.shape == (B, H)
