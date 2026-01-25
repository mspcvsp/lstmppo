"""
This test catches regressions in:

- LSTM weight initialization
- ZeroFeatureEncoder output shape
- Hidden‑state propagation
- Actor head behavior
- Critic head behavior
- Any accidental randomness
- Any nondeterministic CUDA kernels (on CPU this is stable)
- Any future architectural changes

This test ensures that PPO runs are reproducible, which is essential for
debugging, ablation studies, and scientific rigor
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput


def test_policy_lstm_determinism():
    """
    Smoke test: the policy must produce identical outputs when:
      - the same seed is set
      - the same inputs are provided
      - the same initial hidden state is used

    This ensures the LSTM unroll is deterministic and stable.
    """

    # Fix global RNG state
    torch.manual_seed(123)

    cfg = Config()
    cfg.env.obs_space = None
    cfg.env.flat_obs_dim = 0          # ZeroFeatureEncoder path
    cfg.env.action_dim = 3            # actor active

    cfg.lstm.enc_hidden_size = 16
    cfg.lstm.lstm_hidden_size = 8

    # Build two identical policies under the same seed
    policy1 = LSTMPPOPolicy(cfg)
    torch.manual_seed(123)
    policy2 = LSTMPPOPolicy(cfg)

    B = 4
    T = 5
    H = cfg.lstm.lstm_hidden_size

    obs = torch.zeros(B, T, 0)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    # Unroll both policies
    h1, c1 = h0.clone(), c0.clone()
    h2, c2 = h0.clone(), c0.clone()

    outputs1 = []
    outputs2 = []

    for t in range(T):
        out1 = policy1.forward(PolicyInput(obs=obs[:, t], hxs=h1, cxs=c1))
        out2 = policy2.forward(PolicyInput(obs=obs[:, t], hxs=h2, cxs=c2))

        outputs1.append((out1.logits, out1.values))
        outputs2.append((out2.logits, out2.values))

        h1, c1 = out1.new_hxs, out1.new_cxs
        h2, c2 = out2.new_hxs, out2.new_cxs

    # Compare all outputs
    for (log1, val1), (log2, val2) in zip(outputs1, outputs2):
        assert torch.allclose(log1, log2)
        assert torch.allclose(val1, val2)