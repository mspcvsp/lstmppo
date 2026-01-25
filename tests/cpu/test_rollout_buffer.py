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
from lstmppo.types import Config, initialize_config
from lstmppo.types import RolloutStep, LSTMGates


def _make_buffer():
    cfg = initialize_config(Config())
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.trainer.cuda else "cpu"
    )
    return cfg, device, RecurrentRolloutBuffer(cfg, device)


def test_buffer_initialization_shapes():
    cfg, _, buf = _make_buffer()

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs
    D = cfg.env.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    assert buf.obs.shape == (T, B, D)
    assert buf.actions.shape == (T, B, 1)
    assert buf.rewards.shape == (T, B)
    assert buf.values.shape == (T, B)
    assert buf.logprobs.shape == (T, B)
    assert buf.terminated.shape == (T, B)
    assert buf.truncated.shape == (T, B)
    assert buf.hxs.shape == (T, B, H)
    assert buf.cxs.shape == (T, B, H)


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
        assert t.device.type == device.type
        assert t.dtype in (torch.float32, torch.bool, torch.int64)


def test_add_increments_pointer_and_writes_data():
    
    cfg, _, buf = _make_buffer()

    B = cfg.env.num_envs
    D = cfg.env.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    obs = torch.randn(B, D)
    act = torch.randn(B)
    rew = torch.randn(B)
    val = torch.randn(B)
    logp = torch.randn(B)
    term = torch.zeros(B, dtype=torch.bool)
    trunc = torch.zeros(B, dtype=torch.bool)
    hxs = torch.randn(B, H)
    cxs = torch.randn(B, H)

    gates = LSTMGates(
        i_gates=hxs, f_gates=hxs, g_gates=hxs,
        o_gates=hxs, c_gates=hxs, h_gates=hxs,
    )

    step = RolloutStep(
        obs=obs,
        actions=act,
        rewards=rew,
        values=val,
        logprobs=logp,
        terminated=term,
        truncated=trunc,
        hxs=hxs,
        cxs=cxs,
        gates=gates,
    )

    buf.add(step)

    assert buf.step == 1
    assert torch.allclose(buf.obs[0], obs)
    assert torch.allclose(buf.actions[0].squeeze(-1), act)


def test_fill_buffer_reaches_full_pointer():

    cfg, _, buf = _make_buffer()

    B = cfg.env.num_envs
    D = cfg.env.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    for _ in range(cfg.trainer.rollout_steps):

        obs = torch.randn(B, D)
        act = torch.randn(B)
        rew = torch.randn(B)
        val = torch.randn(B)
        logp = torch.randn(B)
        term = torch.zeros(B, dtype=torch.bool)
        trunc = torch.zeros(B, dtype=torch.bool)
        hxs = torch.randn(B, H)
        cxs = torch.randn(B, H)

        gates = LSTMGates(
            i_gates=hxs, f_gates=hxs, g_gates=hxs,
            o_gates=hxs, c_gates=hxs, h_gates=hxs,
        )

        step = RolloutStep(
            obs=obs,
            actions=act,
            rewards=rew,
            values=val,
            logprobs=logp,
            terminated=term,
            truncated=trunc,
            hxs=hxs,
            cxs=cxs,
            gates=gates,
        )

        buf.add(step)

    assert buf.step == cfg.trainer.rollout_steps


def test_mask_logic_cpu():

    _, _, buf = _make_buffer()

    # manually set termination/truncation
    buf.terminated[0] = True
    buf.truncated[1] = True
    mask = buf.mask

    # mask is (T, B) not scalar
    assert torch.all(buf.mask[0] == 0)
    assert torch.all(buf.mask[1] == 0)
    assert torch.all(buf.mask[2] == 1)


def test_reset_clears_state():
    
    _, _, buf = _make_buffer()

    buf.terminated[0] = True
    buf.truncated[0] = True

    buf.reset()

    assert buf.step == 0
    assert torch.all(buf.terminated == 0)
    assert torch.all(buf.truncated == 0)
    assert torch.all(buf.mask == 1)


def test_rollout_step_structure():
    cfg, _, buf = _make_buffer()

    B = cfg.env.num_envs
    D = cfg.obs_dim
    H = cfg.lstm.lstm_hidden_size

    obs = torch.randn(B, D)
    act = torch.randn(B)
    rew = torch.randn(B)
    val = torch.randn(B)
    logp = torch.randn(B)
    term = torch.zeros(B, dtype=torch.bool)
    trunc = torch.zeros(B, dtype=torch.bool)
    hxs = torch.randn(B, H)
    cxs = torch.randn(B, H)
    gates = LSTMGates(
        i_gates=hxs, f_gates=hxs, g_gates=hxs,
        o_gates=hxs, c_gates=hxs, h_gates=hxs,
    )

    step = RolloutStep(
        obs=obs,
        actions=act,
        rewards=rew,
        values=val,
        logprobs=logp,
        terminated=term,
        truncated=trunc,
        hxs=hxs,
        cxs=cxs,
        gates=gates,
    )

    buf.add(step)

    step = buf.get_step(0)
    assert isinstance(step, RolloutStep)

    assert step.obs.shape == (cfg.obs_dim,)
    assert step.actions.shape == (cfg.env.action_dim,)
    assert isinstance(step.reward, float)
    assert isinstance(step.value, float)
    assert isinstance(step.logprob, float)