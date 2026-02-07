import torch

from lstmppo.buffer import RecurrentRolloutBuffer
from tests.helpers.fake_state import FakeState


def test_aux_targets_alignment(fake_state: FakeState):
    # Create fake rollout
    T, B, obs_dim = 5, 2, fake_state.flat_obs_dim

    buf = RecurrentRolloutBuffer(fake_state, device="cpu")

    # Fill obs with increasing integers so alignment is obvious
    for t in range(T):
        buf.obs[t] = torch.full((B, obs_dim), float(t))
        buf.rewards[t] = t * 10.0

    batches = list(buf.get_recurrent_minibatches())
    batch = batches[0]

    # next_obs[t] = obs[t+1]
    assert torch.allclose(batch.next_obs[:-1], batch.obs[1:])

    # final timestep padded
    assert torch.all(batch.next_obs[-1] == 0)

    # next_rewards[t] = rewards[t]
    assert torch.allclose(batch.next_rewards, buf.rewards)
