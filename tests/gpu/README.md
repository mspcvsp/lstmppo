📘 Recurrent PPO Test Suite — GPU
This directory contains the research‑grade validation suite for the recurrent PPO implementation.
These tests enforce the mathematical and architectural invariants required for:
• 	deterministic LSTM state‑flow
• 	correct TBPTT behavior
• 	reproducible rollouts
• 	stable per‑unit diagnostics (drift, saturation, entropy)
• 	correct hidden‑state alignment
• 	correct masking semantics
• 	shape invariants for all recurrent tensors
The suite is intentionally strict. If any of these tests fail, the recurrent pipeline is no longer guaranteed to be correct.

📁 File Overview

Validates the core recurrent PPO invariants:
• 	TBPTT determinism
Ensures that slicing the sequence into TBPTT chunks produces identical results to full‑sequence evaluation.
• 	Rollout replay determinism
Ensures that given identical , the policy produces identical .
• 	Hidden‑state alignment
Ensures that the buffer stores the pre‑step LSTM state  used to generate the action at time .
• 	Mask correctness
Ensures that terminated/truncated environments do not leak hidden state into the next episode.
These tests guarantee that the recurrent core is mathematically sound and reproducible.


Validates the per‑unit LSTM diagnostics:
• 	Gate means (, , , )
• 	Per‑unit drift (, , etc.)
• 	Gate saturation (sigmoid/tanh saturation fractions)
• 	Gate entropy (per‑unit entropy of gate activations)
• 	Replay determinism for diagnostics
• 	Mask‑aware drift computation
• 	No NaNs / no shape mismatches
These tests ensure that the diagnostics pipeline is stable, interpretable, and mathematically correct.


(Your new file — protects the most fragile invariants.)
This file contains micro‑tests that catch common regressions instantly:
• 	LSTM state shape invariant
Ensures  and  are always .
• 	State‑flow initialization invariant
Ensures  is self‑contained and initializes LSTM states even on a fresh trainer.
These tests prevent subtle shape bugs and initialization errors from creeping back in.

🧠 Why these tests matter
Recurrent PPO is extremely sensitive to:
• 	hidden‑state alignment
• 	deterministic transitions
• 	correct masking
• 	correct TBPTT slicing
• 	stable per‑unit metrics
A single shape mismatch or incorrect state carry‑over can silently corrupt training.
This suite ensures that every rollout, every update, and every diagnostic is mathematically correct.

🧪 Running the suite
From the project root: `pytest -q tests/gpu/`
To run a single file:  `pytest tests/gpu/test_recurrent_core.py -q`
To run a single test: `pytest tests/gpu/test_recurrent_core.py::test_rollout_replay_determinism -q`


🏁 Contributing Guidelines
When modifying:
• 	the LSTM core
• 	the rollout buffer
• 	the env wrapper
• 	TBPTT slicing
• 	diagnostics computation
Run this suite before committing.
If a test fails, it means a core invariant has been broken — fix the invariant, not the test.
