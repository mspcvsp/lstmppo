import torch
import pytest
from lstmppo.trainer import LSTMPPOTrainer

# ------------------------------------------------------------
# Deterministic trainer fixture
# ------------------------------------------------------------
@pytest.fixture
def deterministic_trainer():
    trainer = LSTMPPOTrainer.for_validation()

    # Hard determinism
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer.policy.eval()
    trainer.state.cfg.trainer.debug_mode = True

    return trainer


# ------------------------------------------------------------
# 1. TBPTT determinism test
# ------------------------------------------------------------
def test_tbptt_determinism(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.collect_rollout()

    # This will assert internally
    trainer.validate_tbptt()


# ------------------------------------------------------------
# 2. Rollout → Replay determinism test
# ------------------------------------------------------------
def test_rollout_replay_determinism(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.validate_lstm_state_flow()  # asserts internally


# ------------------------------------------------------------
# 3. Hidden-state alignment test
# ------------------------------------------------------------
def test_hidden_state_alignment(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.collect_rollout()

    batch = next(trainer.buffer.get_recurrent_minibatches())

    # Hidden states must be (T, B, H)
    assert batch.hxs.shape[0] == trainer.rollout_steps
    assert batch.hxs.shape[1] == trainer.num_envs

    # 2. Hidden states must be PRE-STEP states
    #    So hxs[t] must equal the state used to compute value/logprob at t.
    #    We validate this by ensuring the state-flow validator passes.
    trainer.validate_lstm_state_flow()

# Shape correctness for all per‑unit metrics
def test_unit_metrics_shapes(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.collect_rollout()

    diag = trainer.compute_lstm_unit_diagnostics_from_rollout()
    H = diag.hidden_size

    for name in [
        "i_mean", "f_mean", "g_mean", "o_mean",
        "h_norm", "c_norm",
        "i_drift", "f_drift", "g_drift", "o_drift",
        "h_drift", "c_drift",
    ]:
        t = getattr(diag, name)
        assert t.shape == (H,)

def test_unit_metrics_mask_behavior(deterministic_trainer):
    trainer = deterministic_trainer

    trainer.collect_rollout()
    full = trainer.compute_lstm_unit_diagnostics_from_rollout()

    # Mask out half the rollout
    T = trainer.rollout_steps
    mask = torch.zeros(T, trainer.num_envs).to(trainer.device)
    mask[: T // 2] = 1.0

    eval_output = trainer.replay_policy_on_rollout()
    half = trainer.compute_lstm_unit_diagnostics(eval_output, mask)

    assert not torch.allclose(full.i_mean, half.i_mean)

def test_unit_metrics_drift_correctness(deterministic_trainer):
    trainer = deterministic_trainer

    trainer.collect_rollout()
    first = trainer.compute_lstm_unit_diagnostics_from_rollout()

    trainer.collect_rollout()
    second = trainer.compute_lstm_unit_diagnostics_from_rollout()

    assert torch.allclose(second.i_drift, second.i_mean - first.i_mean)
    assert torch.allclose(second.h_drift, second.h_norm - first.h_norm)

# Test: No NaNs under extreme activations
def test_unit_metrics_no_nans(deterministic_trainer):
    trainer = deterministic_trainer

    with torch.no_grad():
        for p in trainer.policy.parameters():
            p.mul_(50.0)

    trainer.collect_rollout()
    diag = trainer.compute_lstm_unit_diagnostics_from_rollout()

    for name, val in diag.__dict__.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all()
        elif isinstance(val, dict):
            for v in val.values():
                assert torch.isfinite(v).all()

def test_unit_metrics_replay_determinism(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    diag1 = trainer.compute_lstm_unit_diagnostics_from_rollout()
    diag2 = trainer.compute_lstm_unit_diagnostics_from_rollout()

    for name, v1 in diag1.__dict__.items():
        v2 = getattr(diag2, name)

        # Skip non-tensor fields
        if not isinstance(v1, torch.Tensor):
            continue

        # Handle nested dataclasses (saturation, entropy)
        if hasattr(v1, "__dict__"):
            for k, t1 in v1.__dict__.items():
                t2 = getattr(v2, k)
                if isinstance(t1, torch.Tensor):
                    assert torch.allclose(t1, t2, atol=1e-8)
            continue

        # Normal tensor field
        assert torch.allclose(v1, v2, atol=1e-8)

"""
mask influences:

- GAE bootstrapping
- return normalization
- advantage normalization
- PPO loss masking
- entropy masking
- gate entropy/saturation masking
- drift masking
- TBPTT chunk masking
- diagnostics masking
- replay determinism

If the mask is wrong, everything downstream becomes subtly wrong, and the
bugs are extremely hard to detect.
"""
def test_mask_correctness(deterministic_trainer):
    
    trainer = deterministic_trainer
    trainer.collect_rollout()

    buf = trainer.buffer

    # 1. Mask shape
    assert buf.mask.shape == buf.terminated.shape

    # 2. Mask = 1 - (terminated OR truncated)
    expected = 1.0 - (buf.terminated | buf.truncated).float()
    assert torch.allclose(buf.mask, expected)

    # 3. Mask must be 0 where terminated or truncated
    assert torch.all(buf.mask[buf.terminated] == 0)
    assert torch.all(buf.mask[buf.truncated] == 0)

    # 4. Mask must be 1 where alive
    alive = ~(buf.terminated | buf.truncated)
    assert torch.all(buf.mask[alive] == 1)
