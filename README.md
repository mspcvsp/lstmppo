## TL;DR — RuntimeEnvInfo

`RuntimeEnvInfo` is the *only* authoritative source of environment metadata.
The trainer extracts metadata once from the real Gym environment and stores it here.
Everything else — policy, buffer, diagnostics, tests — must read from `env_info`.

**Why:**
- Prevents stale or duplicated dimensions
- Eliminates encoder/LSTM/action‑space mismatches
- Simplifies tests
- Future‑proofs multi‑env setups

**TrainerState only stores training‑time scalars.**
It exposes compatibility aliases (`flat_obs_dim`, `action_dim`) that forward to `env_info`.

**Tests may override `env_info` directly** without constructing a real environment.
