# Invariant Hierarchy

LSTM‑PPO Diagnostics Suite
│
├── Gate‑Level Invariants
│     ├── Shapes
│     ├── Detachment
│     ├── Saturation (sigmoid/tanh/combined)
│     ├── Entropy
│     ├── Time‑major consistency
│     └── Gate–cell correlation
│
├── Drift & Stability
│     ├── Hidden drift
│     ├── Cell drift
│     ├── Drift ratio
│     ├── Drift growth (expected)
│     ├── Drift smoothness
│     ├── Drift variance
│     ├── Drift–saturation coupling
│     ├── Long‑horizon stability
│     └── Cell saturation
│
├── Policy‑Level
│     ├── Shapes
│     ├── Determinism
│     ├── Minibatch consistency
│     ├── LSTM‑only mode
│     ├── Encoder identity
│     └── Actor identity
│
├── PPO‑Level
│     ├── Loss structure
│     ├── KL behavior
│     ├── KL watchdog
│     └── Value monotonicity
│
└── Infrastructure
      ├── Config
      ├── GAE
      ├── Rollout buffer
      └── AR/TAR