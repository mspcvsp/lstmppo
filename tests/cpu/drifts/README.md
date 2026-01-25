# Drift & Stability Invariants

These tests validate long‑horizon behavior of the LSTM:

- Hidden‑state drift accumulation
- Cell‑state drift accumulation
- Drift growth rate (in expectation)
- Drift smoothness across horizons
- Drift variance stability
- Drift–saturation coupling
- Hidden vs cell drift ratio
- Long‑horizon boundedness
- Cell‑state saturation under extreme inputs

These invariants ensure the LSTM is stable, non‑explosive, and mathematically interpretable.