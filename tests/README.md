# Test Suite Overview

This directory contains all automated tests for the LSTM‑PPO project.
The suite is organized by subsystem to ensure clarity, maintainability, and regression‑proof behavior.

## Structure

- `cpu/` — deterministic CPU‑safe tests validating architecture, drift, PPO logic, and infrastructure.
- `gpu/` — optional GPU‑accelerated tests (if present).
- `integration/` — end‑to‑end training tests (if present).

Each subdirectory contains its own README describing the invariants it enforces.

## Philosophy

This test suite is built around three principles:

1. **Invariants over exact values**
   Tests enforce relationships (monotonicity, boundedness, consistency), not brittle numeric equality.

2. **Expectation‑based reasoning**
   Where drift or saturation is stochastic, tests use averages and tolerances.

3. **Interpretability as a first‑class goal**
   Gate dynamics, drift, and saturation are validated to ensure the LSTM remains transparent and debuggable.

# Test Architecture Overview

This test suite uses a set of reusable helpers to ensure that all tests
share the same invariants as the real PPO/LSTM training pipeline.

## Helpers

### FakeState
Located in `tests/helpers/fake_state.py`.

Provides a minimal, structurally correct TrainerState implementation
used by:
- LSTMPPOPolicy
- RecurrentRolloutBuffer
- TBPTT chunking
- Aux prediction heads

### FakePolicy
Located in `tests/helpers/fake_policy.py`.

Constructs a valid LSTMPPOPolicy using FakeState.

### FakeRolloutBuilder
Located in `tests/helpers/fake_rollout.py`.

Builds aligned rollouts with:
- obs
- next_obs
- actions
- rewards
- masks
- optional LSTM hidden states

### FakeBufferLoader
Loads a FakeRollout into a RecurrentRolloutBuffer.

### FakeBatchBuilder
Creates minibatches directly from a FakeRollout.

## Why this architecture?

- Guarantees alignment invariants
- Eliminates boilerplate
- Makes tests easier to read and write
- Ensures future changes to rollout/buffer semantics only require updating helpers

Run all tests:
   pytest -q

Run only CPU tests:
   pytest tests/cpu -q

Run only GPU tests:
   pytest tests/gpu -q
