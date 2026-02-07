from lstmppo.types import PolicyEvalInput
from tests.helpers.fake_batch import make_fake_batch
from tests.helpers.fake_policy import make_fake_policy
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import TrainerStateProtocol


def test_aux_gradients(fake_state: TrainerStateProtocol):
    policy = make_fake_policy()

    cfg = fake_state.cfg.buffer_config
    T = cfg.rollout_steps
    B = cfg.num_envs
    D = fake_state.flat_obs_dim

    rollout = FakeRolloutBuilder(T=T, B=B, obs_dim=D).build()
    batch = make_fake_batch(fake_state, rollout)

    out = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=batch.obs,
            hxs=batch.hxs[0],  # [B, H] ✅
            cxs=batch.cxs[0],  # [B, H] ✅
            actions=batch.actions,
        )
    )

    assert out.pred_obs is not None
    assert out.pred_raw is not None
    assert out.pred_obs.requires_grad
    assert out.pred_raw.requires_grad
