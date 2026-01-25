# Gate‑Level Invariants

These tests validate the internal gate dynamics of the LSTM:

- Gate tensor shapes
- Gate detachment from autograd
- Sigmoid/tanh saturation behavior
- Combined saturation metrics
- Gate entropy behavior
- Time‑major vs batch‑major consistency
- Forget‑gate–to–cell‑magnitude correlation

These invariants ensure the LSTM remains interpretable and diagnostically transparent.