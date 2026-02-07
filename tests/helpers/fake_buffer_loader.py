from lstmppo.buffer import RecurrentRolloutBuffer
from tests.helpers.fake_rollout import FakeRollout
from tests.helpers.fake_state import TrainerStateProtocol


def load_rollout_into_buffer(
    state: TrainerStateProtocol,
    rollout: FakeRollout,
    *,
    device: str = "cpu",
) -> RecurrentRolloutBuffer:
    buf = RecurrentRolloutBuffer(state, device=device)

    # obs: [T, B, D]
    assert buf.obs.shape == rollout.obs.shape
    buf.obs.copy_(rollout.obs)

    # actions: buffer [T, B, 1], rollout [T, B]
    assert rollout.actions.dim() == 2
    assert buf.actions.shape == (rollout.actions.shape[0], rollout.actions.shape[1], 1)
    buf.actions.copy_(rollout.actions.unsqueeze(-1))

    # rewards: [T, B]
    assert buf.rewards.shape == rollout.rewards.shape
    buf.rewards.copy_(rollout.rewards)

    assert buf.masks is not None
    assert buf.masks.shape == rollout.masks.shape
    buf.masks.copy_(rollout.masks)

    # optional hidden states
    if rollout.h0 is not None:
        buf.hxs[0].copy_(rollout.h0)
    if rollout.c0 is not None:
        buf.cxs[0].copy_(rollout.c0)

    return buf
