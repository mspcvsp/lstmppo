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
    obs = torch.zeros(T, 1, obs_dim, device=device)

    # Reset hidden state
    h, c = policy.lstm.initial_state(batch_size=1, device=device)

    drift = torch.zeros(T, device=device)

    prev_h = h.clone()

    for t in range(T):
        encoded = policy.encoder(obs[t])
        h, c = policy.lstm(encoded, (h, c))

        # Compute drift magnitude
        diff = h - prev_h
        drift[t] = diff.norm(p=2)

        prev_h = h.clone()

    return drift
