# Dreamer‑V3 World Model — Gradient Flow Overview

This document explains how gradients flow through the Dreamer‑V3 world model during training, including how the observation encoder (MLP for PopGym/CAGE2, CNN for Crafter) is optimized end‑to‑end. Dreamer‑V3 trains one unified world model, consisting of:

- ObsEncoder (MLP or CNN)
- RSSMCore (deterministic transition model)
- Prior / Posterior (factored discrete latent distributions)
- ObsDecoder (reconstructs observations)
- RewardHead (distributional symlog reward)
- ContinueHead (distributional continuation)

All components are trained jointly using reconstruction, reward, continuation, and KL losses.

---

## High‑Level Flow

At each environment step:

obs_t → encoder → embed_t
(h_{t-1}, embed_t) → posterior → z_t
(h_{t-1}) → prior → ẑ_t
(h_{t-1}, a_{t-1}) → RSSM → h_t
(h_t, z_t) → decoder → recon_t
(h_t, z_t) → reward_head → reward_logits_t
(h_t, z_t) → continue_head → cont_logits_t
posterior vs prior → KL divergence

The training loop computes:

loss = recon_loss + reward_loss + continue_loss + kl_loss
loss.backward()

Gradients flow backward through all components:

decoder ← RSSM ← encoder
reward_head ← RSSM ← encoder
continue_head ← RSSM ← encoder
posterior ← encoder
prior ← RSSM
KL ← posterior, prior

Thus the encoder (MLP or CNN) is trained end‑to‑end as part of the world model.

---

## Detailed Gradient Flow

### 1. ObsEncoder (MLP or CNN)

The encoder receives raw observations:

- PopGym: one‑hot vectors
- CAGE2: flattened symbolic dicts
- Crafter: 64×64×3 RGB images

It outputs a continuous embedding embed_t.

Gradients reach the encoder from:

- posterior (via q(z_t | h_{t-1}, embed_t))
- decoder reconstruction
- reward head
- continue head
- KL regularization (indirectly via posterior)

Thus the encoder learns features that improve:

- reconstruction accuracy
- latent inference
- reward prediction
- continuation prediction
- KL stability

---

### 2. Posterior

Posterior receives (h_{t-1}, embed_t) and produces:

- discrete latent z_t
- logits/probs for KL computation

Gradients flow into:

- posterior network
- encoder
- RSSM (indirectly through KL)

---

### 3. Prior

Prior receives h_{t-1} and predicts:

- latent distribution ẑ_t
- logits/probs for KL

Gradients flow into:

- prior network
- RSSM

---

### 4. RSSMCore

RSSM receives (h_{t-1}, a_{t-1}) and produces h_t.

Gradients flow into:

- RSSM transition network
- actor (via imagination)
- encoder (indirectly through posterior/decoder)

---

### 5. ObsDecoder

Decoder reconstructs the observation from (h_t, z_t).

Gradients flow into:

- decoder network
- RSSM
- posterior
- encoder

---

### 6. RewardHead / ContinueHead

These predict reward and continuation distributions.

Gradients flow into:

- reward head
- continue head
- RSSM
- posterior
- encoder

---

### 7. KL Divergence

KL is computed between posterior and prior distributions.

Gradients flow into:

- posterior
- prior
- RSSM
- encoder

---

## AMP Autocast (Mixed Precision)

The world‑model training step is already wrapped in:

with torch.cuda.amp.autocast(enabled=cfg.train.amp):

This is correct and matches Dreamer‑V3’s design.

AMP is safe because:

- Encoder + decoder benefit from FP16 speedups
- RSSM, KL, actor, critic remain in FP32
- Determinism tests disable AMP
- Crafter training enables AMP for performance

No further actions are required.

You do not need to modify:

- encoder
- decoder
- RSSM
- KL
- actor
- critic
- replay buffer

AMP is already integrated at the correct level.

---

## Summary

- The encoder (MLP or CNN) is trained end‑to‑end as part of the world model.
- Gradients flow through decoder, posterior, prior, RSSM, reward head, continue head.
- AMP is already correctly applied and requires no additional changes.
- The polymorphic encoder design (MLP for PopGym/CAGE2, CNN for Crafter) fits perfectly into Dreamer‑V3’s gradient flow.
