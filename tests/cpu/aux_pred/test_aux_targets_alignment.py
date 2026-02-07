import torch

from tests.helpers.fake_batch import make_fake_batch
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import TrainerStateProtocol


def test_aux_targets_alignment(fake_state: TrainerStateProtocol):
    cfg = fake_state.cfg.buffer_config
    T = cfg.rollout_steps
    B = cfg.num_envs
    D = fake_state.flat_obs_dim

    # Build deterministic rollout
    rollout = FakeRolloutBuilder(T=T, B=B, obs_dim=D).with_pattern("range").build()

    batch = make_fake_batch(fake_state, rollout)

    # next_obs[t] = obs[t+1]
    assert torch.allclose(batch.next_obs[:-1], batch.obs[1:])

    # final timestep padded
    assert torch.all(batch.next_obs[-1] == 0)

    # next_rewards[t] = rewards[t]
    assert torch.allclose(batch.next_rewards, rollout.rewards)
