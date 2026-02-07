from tests.helpers.fake_buffer_loader import load_rollout_into_buffer
from tests.helpers.fake_rollout import FakeRollout
from tests.helpers.fake_state import TrainerStateProtocol


def make_fake_batch(
    state: TrainerStateProtocol,
    rollout: FakeRollout,
    *,
    device="cpu",
):
    buf = load_rollout_into_buffer(state, rollout, device=device)
    batch = next(buf.get_recurrent_minibatches())
    return batch
