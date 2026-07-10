import numpy as np
import torch

from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.utils.types import make_default_config


def test_replay_buffer_determinism():
    torch.manual_seed(0)
    np.random.seed(0)

    capacity = 10_000
    obs_dim = 8
    action_dim = 1
    device = torch.device("cpu")
    seq_len = 5

    cfg = make_default_config()
    cfg.train.replay_capacity = capacity
    cfg.train.seq_len = seq_len

    rb = ReplayBuffer(cfg=cfg.train, obs_dim=obs_dim, action_dim=action_dim, device=device)

    # Seed BEFORE filling the buffer
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Fill with deterministic episodes ---
    # Two episodes of length 10 each
    for _ in range(2):
        for t in range(10):
            obs = torch.randn(1, obs_dim)
            action = torch.randint(0, 2, (1,))
            reward = torch.randn(1)
            done = torch.tensor([1.0 if t == 9 else 0.0])

            rb.add(obs, action, reward, done)

    # --- Sample twice with same seed ---
    batch1 = rb.sample(batch_size=4, seed=123)
    batch2 = rb.sample(batch_size=4, seed=123)

    # --- Compare ---
    for key in batch1.keys():
        assert torch.allclose(batch1[key], batch2[key]), f"Mismatch in field: {key}"
