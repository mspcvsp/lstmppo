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
    obs_dim = policy.obs_dim
    obs = torch.zeros(T, 1, obs_dim, device=device)  # (T, B=1, obs_dim)

    # Reset hidden state: (B, H)
    h, c = policy.lstm.initial_state(batch_size=1, device=device)

    drift = torch.zeros(T, device=device)

    prev_h = h.clone()  # (B, H)

    for t in range(T):
        # encoded: (B, enc_dim)
        encoded = policy.encoder(obs[t])  # obs[t]: (1, obs_dim) → encoded: (1, enc_dim)

        # GateLSTMCell expects x: (B, input_size), h,c: (B, H)
        h, c = policy.lstm(encoded, (h, c))  # returns (h_new, c_new), both (B, H)

        # L2 drift per batch element, then average over batch (B=1 here)
        drift[t] = (h - prev_h).norm(dim=-1).mean()

        prev_h = h.clone()

    return drift
