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

Run all tests:
   pytest -q

Run only CPU tests:
   pytest tests/cpu -q

Run only GPU tests:
   pytest tests/gpu -q