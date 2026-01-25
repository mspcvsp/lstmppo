# Policy‑Level Invariants

These tests validate the behavior of the policy network:

- Output shapes
- Minibatch consistency
- Determinism in eval mode
- LSTM‑only mode correctness
- Encoder identity behavior
- Actor identity behavior
- Combined identity path

These invariants ensure the policy is stable, predictable, and regression‑proof.