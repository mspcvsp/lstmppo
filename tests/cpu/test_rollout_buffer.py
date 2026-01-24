"""
CPU-side validation for RecurrentRolloutBuffer.

This suite catches:
- shape regressions
- pointer/indexing bugs
- mask logic errors
- device/dtype drift
- incorrect initialization/reset behavior
- incorrect RolloutStep structure
"""
import torch
from lstmppo.buffer import RecurrentRolloutBuffer
from lstmppo.types import Config, initialize_config, RolloutStep


def _make_buffer():
    cfg = initialize_config(Config())
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.trainer.cuda else "cpu"
    )
    return cfg, device, RecurrentRolloutBuffer(cfg, device)


def test_buffer_initialization_shapes():
    cfg, _, buf = _make_buffer()

    T = cfg.trainer.rollout_steps
    obs_dim = cfg.env.obs_dim
    act_dim = cfg.env.action_dim

    assert buf.ptr == 0
    assert buf.obs.shape == (T, obs_dim)
    assert buf.actions.shape == (T, act_dim)
    assert buf.rewards.shape == (T,)
    assert buf.values.shape == (T,)
    assert buf.logprobs.shape == (T,)
    assert buf.terminated.shape == (T,)
    assert buf.truncated.shape == (T,)
    assert buf.mask.shape == (T,)


def test_buffer_device_and_dtype():
    _, device, buf = _make_buffer()

    for t in [
        buf.obs,
        buf.actions,
        buf.rewards,
        buf.values,
        buf.logprobs,
        buf.terminated,
        buf.truncated,
        buf.mask,
    ]:
        assert t.device == device
        assert t.dtype in (torch.float32, torch.bool, torch.int64)


def test_add_increments_pointer_and_writes_data():
    cfg, device, buf = _make_buffer()

    obs = torch.randn(cfg.env.obs_dim)
    act = torch.randn(cfg.env.action_dim)
    rew = 1.0
    done = False
    val = 0.5
    logp = -0.1

    buf.add(obs, act, rew, done, val, logp)

    assert buf.ptr == 1
    assert torch.allclose(buf.obs[0], obs.to(device))
    assert torch.allclose(buf.actions[0], act.to(device))
    assert buf.rewards[0].item() == rew
    assert buf.values[0].item() == val
    assert buf.logprobs[0].item() == logp
    assert buf.terminated[0].item() == 0
    assert buf.truncated[0].item() == 0


def test_fill_buffer_reaches_full_pointer():

    cfg, _, buf = _make_buffer()

    obs = torch.randn(cfg.env.obs_dim)
    act = torch.randn(cfg.env.action_dim)

    for _ in range(cfg.trainer.rollout_steps):
        buf.add(obs, act, 1.0, False, 0.5, -0.1)

    assert buf.step == cfg.trainer.rollout_steps


def test_mask_logic_cpu():
    _, _, buf = _make_buffer()

    # manually set termination/truncation
    buf.terminated[0] = True
    buf.truncated[1] = True

    buf.compute_mask()

    assert buf.mask[0].item() == 0
    assert buf.mask[1].item() == 0
    assert buf.mask[2].item() == 1  # untouched index


def test_reset_clears_state():
    cfg, _, buf = _make_buffer()

    # fill some data
    obs = torch.randn(cfg.env.obs_dim)
    act = torch.randn(cfg.env.action_dim)
    buf.add(obs, act, 1.0, False, 0.5, -0.1)

    buf.reset()

    assert buf.ptr == 0
    assert torch.all(buf.terminated == 0)
    assert torch.all(buf.truncated == 0)
    assert torch.all(buf.mask == 1)


def test_rollout_step_structure():
    cfg, _, buf = _make_buffer()

    obs = torch.randn(cfg.env.obs_dim)
    act = torch.randn(cfg.env.action_dim)
    buf.add(obs, act, 1.0, False, 0.5, -0.1)

    step = buf.get_step(0)
    assert isinstance(step, RolloutStep)

    assert step.obs.shape == (cfg.env.obs_dim,)
    assert step.action.shape == (cfg.env.action_dim,)
    assert isinstance(step.reward, float)
    assert isinstance(step.value, float)
    assert isinstance(step.logprob, float)