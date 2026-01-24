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
    mask = torch.zeros(T, trainer.num_envs)
    mask[: T // 2] = 1.0

    eval_output = trainer.replay_policy_on_rollout()
    half = trainer.compute_lstm_unit_diagnostics(eval_output, mask)

    assert not torch.allclose(full.i_mean, half.i_mean)
