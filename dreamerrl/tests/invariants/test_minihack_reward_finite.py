import pytest
import torch


@pytest.mark.invariants
@pytest.mark.minihack_invariants
def test_reward_finite(minihack_env):
    minihack_env.reset()
    actions = torch.zeros(minihack_env.batch_size, dtype=torch.long)
    out = minihack_env.step(actions)

    # Reward is finite and scalar per env
    assert out["reward"].shape == (minihack_env.batch_size,)
    assert torch.isfinite(out["reward"]).all()
