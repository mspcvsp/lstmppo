import torch


@torch.no_grad()
def compute_drift_sequence(policy, T: int = 64):
    """
    Compute the hidden-state drift sequence for a single rollout of length T.

    Drift is defined as the L2 norm of the difference between consecutive
    hidden states: ||h_t - h_{t-1}||_2.

    Args:
        policy: LSTMPPOPolicy instance with .lstm and .encoder modules.
        T: sequence length.

    Returns:
        drift: tensor of shape (T,) containing drift magnitudes.
    """

    device = next(policy.parameters()).device

    # Create a dummy observation sequence
    # The encoder is the source of truth for obs_dim
    obs_dim = policy.encoder[0].in_features

    obs = torch.zeros(T, 1, obs_dim, device=device)  # (T, B=1, obs_dim)

    # Reset hidden state: (B, H)
    h, c = policy.lstm.initial_state(batch_size=1, device=device)

    drift = torch.zeros(T, device=device)

    prev_h = h.clone()  # (B, H)

    for t in range(T):
        # encoded: (B, enc_dim)
        encoded = policy.encoder(obs[t])  # obs[t]: (1, obs_dim) → encoded: (1, enc_dim)

        # GateLSTMCell returns (h_new, c_new, extra)
        h, c, _ = policy.lstm(encoded, (h, c))

        # L2 drift per batch element, then average over batch (B=1 here)
        drift[t] = (h - prev_h).norm(dim=-1).mean()

        prev_h = h.clone()

    return drift
