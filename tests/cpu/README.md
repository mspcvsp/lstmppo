# CPU Test Suite

This directory contains all CPU‑safe, deterministic tests for the LSTM‑PPO architecture.  
These tests validate:

- LSTM gate correctness
- Drift and stability behavior
- Policy forward‑pass invariants
- PPO algorithm correctness
- Infrastructure components (GAE, buffers, config)

The suite is divided into five submodules:

- `gates/` — gate‑level invariants
- `drift/` — drift, saturation, and long‑horizon stability
- `policy/` — policy‑level invariants
- `ppo/` — PPO algorithm invariants
- `infra/` — supporting infrastructure

Each subfolder includes a README describing its invariants.