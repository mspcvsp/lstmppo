import torch

from lstmppo.buffer import RecurrentRolloutBuffer
from tests.helpers.fake_state import FakeState


def test_aux_targets_tbptt_chunks(fake_state: FakeState):
    T = 8
    K = 3  # chunk length

    buf = RecurrentRolloutBuffer(fake_state, device="cpu")

    # Fill obs with t for easy checking
    for t in range(T):
        buf.obs[t] = t
        buf.rewards[t] = t

    batch = next(buf.get_recurrent_minibatches())
    chunks = list(batch.iter_chunks(K))

    for idx, mb in enumerate(chunks):
        # next_obs alignment inside chunk
        assert torch.allclose(mb.next_obs[:-1], mb.obs[1:])

        # next_rewards alignment
        assert torch.allclose(mb.next_rewards, buf.rewards[mb.t0 : mb.t1])

        # only the final global timestep is padded
        if idx == len(chunks) - 1:
            assert torch.all(mb.next_obs[-1] == 0)
