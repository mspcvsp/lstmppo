import torch

from tests.helpers.fake_buffer_loader import load_rollout_into_buffer
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import TrainerStateProtocol


def test_aux_targets_tbptt_chunks(fake_state: TrainerStateProtocol):
    T = 8
    K = 3
    D = fake_state.flat_obs_dim
    B = fake_state.cfg.buffer_config.num_envs

    rollout = FakeRolloutBuilder(T=T, B=B, obs_dim=D).with_pattern("range").build()

    buf = load_rollout_into_buffer(fake_state, rollout)
    batch = next(buf.get_recurrent_minibatches())
    chunks = list(batch.iter_chunks(K))

    for idx, mb in enumerate(chunks):
        # next_obs alignment inside chunk
        assert torch.allclose(mb.next_obs[:-1], mb.obs[1:])

        # next_rewards alignment
        assert torch.allclose(mb.next_rewards, rollout.rewards[mb.t0 : mb.t1])

        # only the final global timestep is padded
        if idx == len(chunks) - 1:
            assert torch.all(mb.next_obs[-1] == 0)
