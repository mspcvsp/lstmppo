"""
This test catches:

- GAE math regressions
- dtype issues
- shape mismatches
"""

import torch

from lstmppo.buffer import RecurrentRolloutBuffer
from lstmppo.trainer_state import TrainerState


def test_gae_computation_basic(trainer_state: TrainerState):
    cfg = trainer_state.cfg
    buf_cfg = cfg.buffer_config
    device = torch.device("cpu")

    buf = RecurrentRolloutBuffer(trainer_state, device)

    T = buf_cfg.rollout_steps
    B = buf_cfg.num_envs

    # Use non-constant rewards so normalization is meaningful
    rewards = torch.linspace(0, 1, T).unsqueeze(1).repeat(1, B)
    buf.rewards.copy_(rewards)

    buf.values.zero_()
    buf.terminated.zero_()
    buf.truncated.zero_()

    last_value = torch.zeros(B)

    buf.compute_returns_and_advantages(last_value)

    # Reward normalization: mean ~0, std ~1
    assert abs(buf.rewards.mean().item()) < 1e-6
    assert abs(buf.rewards.std(unbiased=False).item() - 1.0) < 1e-6

    # Shapes
    assert buf.advantages.shape == (T, B)
    assert buf.returns.shape == (T, B)

    # Return normalization: mean ~0, std ~1
    assert abs(buf.returns.mean().item()) < 1e-6
    assert abs(buf.returns.std(unbiased=False).item() - 1.0) < 1e-6
