import torch

from lstmppo.buffer import RecurrentRolloutBuffer

from .conftest import fake_state


def test_aux_targets_tbptt_chunks():
    T = 8
    K = 3  # chunk length

    buf = RecurrentRolloutBuffer(fake_state, device="cpu")

    # Fill obs with t for easy checking
    for t in range(T):
        buf.obs[t] = t
        buf.rewards[t] = t

    batch = next(buf.get_recurrent_minibatches())
    chunks = list(batch.iter_chunks(K))

    for mb in chunks:
        # next_obs alignment inside chunk
        assert torch.allclose(mb.next_obs[:-1], mb.obs[1:])

        # final timestep of chunk padded
        assert torch.all(mb.next_obs[-1] == 0)

        # next_rewards alignment
        assert torch.allclose(mb.next_rewards, buf.rewards[mb.t0 : mb.t1])
