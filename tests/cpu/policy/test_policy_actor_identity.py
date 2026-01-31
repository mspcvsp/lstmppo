import torch
import torch.nn as nn

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import Config, PolicyInput


def test_policy_actor_identity_path(trainer_state: TrainerState):
    """
    Smoke test: when action_dim == 0, the policy must:
      - use nn.Identity() as the actor head
      - avoid constructing Linear(H, 0)
      - produce correctly shaped outputs for a dummy forward pass
    """

    cfg = Config()

    # Force actor identity path
    trainer_state.action_dim = 0
    trainer_state.flat_obs_dim = 4  # nonzero so encoder is normal
    cfg.lstm.enc_hidden_size = 16
    cfg.lstm.lstm_hidden_size = 8

    policy = LSTMPPOPolicy(trainer_state)

    # --- Actor must be Identity ---
    assert isinstance(policy.actor, nn.Identity)

    # --- Forward pass must work with action_dim == 0 ---
    B = 4
    H = cfg.lstm.lstm_hidden_size

    obs = torch.zeros(B, trainer_state.flat_obs_dim)
    hxs = torch.zeros(B, H)
    cxs = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=hxs, cxs=cxs))

    # --- Output sanity checks ---
    assert out.logits.shape == (B, 0)
    assert out.values.shape == (B,)
    assert out.new_hxs.shape == (B, H)
    assert out.new_cxs.shape == (B, H)
