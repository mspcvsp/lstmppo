import torch
import pytest
from lstmppo.trainer import LSTMPPOTrainer

pytestmark = pytest.mark.gpu


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


"""
Drift‑trend test suite that:
- validates initialization
- validates update logic
- validates mask behavior
- validates replay determinism
- validates numerical stability
- validates correctness of the drift formula
- detects LSTM collapse or freezing
- detects exploding dynamics
- protects against regressions in gate extraction
"""
def test_drift_zero_on_first_rollout(deterministic_trainer):

    trainer = deterministic_trainer

    trainer.collect_rollout()
    diag = trainer.compute_lstm_unit_diagnostics_from_rollout()

    for name in ["i_drift", "f_drift", "g_drift",
                 "o_drift", "h_drift", "c_drift"]:
        
        drift = getattr(diag, name)
        assert torch.allclose(drift, torch.zeros_like(drift))


def test_drift_changes_over_time(deterministic_trainer):

    trainer = deterministic_trainer

    trainer.collect_rollout()
    d1 = trainer.compute_lstm_unit_diagnostics_from_rollout()

    trainer.collect_rollout()
    d2 = trainer.compute_lstm_unit_diagnostics_from_rollout()

    # At least one drift dimension should change
    diffs = []

    for name in ["i_drift", "f_drift", "g_drift",
                 "o_drift", "h_drift", "c_drift"]:
        
        diffs.append(torch.allclose(getattr(d1, name), getattr(d2, name)))

    assert not all(diffs), "All drift vectors identical across rollouts"


def test_drift_mask_awareness(deterministic_trainer):

    trainer = deterministic_trainer

    trainer.collect_rollout()
    full = trainer.compute_lstm_unit_diagnostics_from_rollout()

    # Mask out half the rollout
    T = trainer.rollout_steps
    mask = torch.zeros(T, trainer.num_envs).to(trainer.device)
    mask[: T // 2] = 1.0

    eval_output = trainer.replay_policy_on_rollout()
    half = trainer.compute_lstm_unit_diagnostics(eval_output, mask)

    # Means must differ → drift must differ
    assert not torch.allclose(full.i_drift, half.i_drift)


def test_drift_replay_determinism(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    d1 = trainer.compute_lstm_unit_diagnostics_from_rollout()
    d2 = trainer.compute_lstm_unit_diagnostics_from_rollout()

    for name in ["i_drift", "f_drift", "g_drift",
                 "o_drift", "h_drift", "c_drift"]:

        assert torch.allclose(getattr(d1, name),
                              getattr(d2, name), atol=1e-8)
        

def test_drift_sanity_bounds(deterministic_trainer):

    trainer = deterministic_trainer

    trainer.collect_rollout()
    trainer.collect_rollout()
    diag = trainer.compute_lstm_unit_diagnostics_from_rollout()

    for name in ["i_drift", "f_drift", "g_drift",
                 "o_drift", "h_drift", "c_drift"]:
        
        drift = getattr(diag, name)
        assert torch.isfinite(drift).all()
        assert drift.abs().max() < 10.0  # generous bound


def test_drift_matches_mean_difference(deterministic_trainer):

    trainer = deterministic_trainer

    trainer.collect_rollout()
    d1 = trainer.compute_lstm_unit_diagnostics_from_rollout()

    trainer.collect_rollout()
    d2 = trainer.compute_lstm_unit_diagnostics_from_rollout()

    assert torch.allclose(d2.i_drift, d2.i_mean - d1.i_mean)
    assert torch.allclose(d2.h_drift, d2.h_norm - d1.h_norm)


"""
Saturation and entropy monotonicity tests are incredibly powerful because
they catch:

- gate collapse
- gate freezing
- numerical instability
- mask‑related regressions
- incorrect vectorized saturation/entropy logic
- incorrect clamping behavior
- incorrect per‑unit aggregation

And they do it without depending on environment dynamics.
---------------------------------------------------------------
What “monotonicity” means for LSTM-PPO

For saturation:
--------------
- If gate activations move closer to saturation (0 or 1), saturation fraction
  must increase.
- If gate activations move away from saturation, saturation fraction must 
  decrease.

For entropy:
-----------
- If gate activations become more extreme (closer to 0 or 1 for sigmoid,
  ±1 for tanh), entropy must decrease.
- If gate activations become more uniform (closer to 0.5 for sigmoid, 0 for
  tanh), entropy must increase.
-----------------------------------------------------------------------
This is a test suite that validates:

- saturation increases when gates saturate
- saturation decreases when gates desaturate
- entropy decreases when gates saturate
- entropy increases when gates desaturate
- sigmoid and tanh behavior both correct
- vectorized masking logic correct
- clamping logic correct
- no numerical instability
- no regression in gate extraction
"""
def test_sigmoid_saturation_monotonicity(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    # Baseline diagnostics
    base = trainer.compute_lstm_unit_diagnostics_from_rollout()
    base_sat = base.saturation

    # Force gates toward saturation by scaling weights
    with torch.no_grad():
        for p in trainer.policy.lstm_cell.parameters():
            p.mul_(3.0)

    trainer.collect_rollout()
    sat = trainer.compute_lstm_unit_diagnostics_from_rollout().saturation

    # Saturation must increase for sigmoid gates
    assert (sat.i_sat_low >= base_sat.i_sat_low).all()
    assert (sat.i_sat_high >= base_sat.i_sat_high).all()
    assert (sat.f_sat_low >= base_sat.f_sat_low).all()
    assert (sat.f_sat_high >= base_sat.f_sat_high).all()
    assert (sat.o_sat_low >= base_sat.o_sat_low).all()
    assert (sat.o_sat_high >= base_sat.o_sat_high).all()


def test_tanh_saturation_monotonicity(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    base = trainer.compute_lstm_unit_diagnostics_from_rollout()
    base_sat = base.saturation

    # Push tanh gates toward ±1
    with torch.no_grad():
        for p in trainer.policy.lstm_cell.parameters():
            p.mul_(3.0)

    trainer.collect_rollout()
    sat = trainer.compute_lstm_unit_diagnostics_from_rollout().saturation

    assert (sat.g_sat >= base_sat.g_sat).all()
    assert (sat.c_sat >= base_sat.c_sat).all()
    assert (sat.h_sat >= base_sat.h_sat).all()


# Entropy must decrease when gates become more extreme.
def test_sigmoid_entropy_monotonicity(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    base = trainer.compute_lstm_unit_diagnostics_from_rollout()
    base_ent = base.entropy

    # Push gates toward 0/1
    with torch.no_grad():
        for p in trainer.policy.lstm_cell.parameters():
            p.mul_(3.0)

    trainer.collect_rollout()
    ent = trainer.compute_lstm_unit_diagnostics_from_rollout().entropy

    assert (ent.i_entropy <= base_ent.i_entropy).all()
    assert (ent.f_entropy <= base_ent.f_entropy).all()
    assert (ent.o_entropy <= base_ent.o_entropy).all()


def test_tanh_entropy_monotonicity(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    base = trainer.compute_lstm_unit_diagnostics_from_rollout()
    base_ent = base.entropy

    # Push tanh gates toward ±1
    with torch.no_grad():
        for p in trainer.policy.lstm_cell.parameters():
            p.mul_(3.0)

    trainer.collect_rollout()
    ent = trainer.compute_lstm_unit_diagnostics_from_rollout().entropy

    assert (ent.g_entropy <= base_ent.g_entropy).all()
    assert (ent.c_entropy <= base_ent.c_entropy).all()
    assert (ent.h_entropy <= base_ent.h_entropy).all()


# Verify entropy increases when gates become more uniform. This is the
# inverse monotonicity test.
def test_entropy_increases_when_gates_uniformize(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.collect_rollout()

    base = trainer.compute_lstm_unit_diagnostics_from_rollout()
    base_ent = base.entropy

    # Push gates toward 0.5 / 0
    with torch.no_grad():
        for p in trainer.policy.lstm_cell.parameters():
            p.mul_(0.1)

    trainer.collect_rollout()
    ent = trainer.compute_lstm_unit_diagnostics_from_rollout().entropy

    assert (ent.i_entropy >= base_ent.i_entropy).all()
    assert (ent.f_entropy >= base_ent.f_entropy).all()
    assert (ent.o_entropy >= base_ent.o_entropy).all()
    assert (ent.g_entropy >= base_ent.g_entropy).all()


# Verify saturation decreases when gates move away from extremes
def test_saturation_decreases_when_gates_uniformize(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    base = trainer.compute_lstm_unit_diagnostics_from_rollout()
    base_sat = base.saturation

    # Push gates toward center
    with torch.no_grad():
        for p in trainer.policy.lstm_cell.parameters():
            p.mul_(0.1)

    trainer.collect_rollout()
    sat = trainer.compute_lstm_unit_diagnostics_from_rollout().saturation

    assert (sat.i_sat_low <= base_sat.i_sat_low).all()
    assert (sat.i_sat_high <= base_sat.i_sat_high).all()
    assert (sat.f_sat_low <= base_sat.f_sat_low).all()
    assert (sat.f_sat_high <= base_sat.f_sat_high).all()
    assert (sat.o_sat_low <= base_sat.o_sat_low).all()
    assert (sat.o_sat_high <= base_sat.o_sat_high).all()
    assert (sat.g_sat <= base_sat.g_sat).all()
    assert (sat.c_sat <= base_sat.c_sat).all()
    assert (sat.h_sat <= base_sat.h_sat).all()