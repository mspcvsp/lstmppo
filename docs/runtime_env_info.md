# Design Note: RuntimeEnvInfo as the Single Source of Truth

## Overview
RuntimeEnvInfo centralizes all environment‑dependent metadata so the trainer, policy, and buffer never rely on stale or duplicated fields. This eliminates entire classes of bugs involving mismatched dimensions, incorrect action‑space assumptions, and inconsistent observation shapes.

## Why This Exists
Historically, environment metadata was scattered across:
- TrainerState
- the policy constructor
- the rollout buffer
- test fixtures
This led to:
- duplicated fields
- stale values
- shape mismatches
- brittle tests
RuntimeEnvInfo fixes this by becoming the only place where environment metadata lives.

## Core Principles
1. Trainer owns the real environment
Only the trainer touches the actual Gym environment.
It extracts metadata once and stores it in RuntimeEnvInfo.
2. TrainerState stores only training‑time scalars
TrainerState should not store:
- observation spaces
- action spaces
- flattened dimensions
- anything derived from the environment

These belong exclusively in RuntimeEnvInfo.
1. RuntimeEnvInfo is authoritative
It contains:
- obs_space: gym.Space | None
- flat_obs_dim: int
- action_dim: int
It is:
- created via RuntimeEnvInfo.from_env(env)
- optionally overridden by tests
- read by the policy, buffer, and diagnostics

1. Compatibility aliases in TrainerState
To avoid breaking older code:
@property
def flat_obs_dim(self): return self.env_info.flat_obs_dim

@property
def action_dim(self): return self.env_info.action_dim


These are aliases, not independent fields.
5. Policy and buffer must always read from env_info
Never from:
- state.flat_obs_dim
- state.action_dim
- state.obs_space
This guarantees consistent dimensions across:
- encoder
- LSTM
- actor head
- critic head
- rollout buffer
6. Tests may set env_info manually
This is intentional and supported:
state.env_info.flat_obs_dim = 4
state.env_info.action_dim = 0


This allows isolated unit tests without constructing a real environment.

Benefits
- No duplicated metadata
- No stale dimensions
- No mismatched encoder/LSTM shapes
- Cleaner tests
- Clearer architecture
- Future‑proof for multi‑env or multi‑space setups
