import torch

from .types import LSTMStates, RecurrentBatch, RolloutStep


class RecurrentRolloutBuffer:
    def __init__(self, state, device):
        self.device = device

        self.cfg = state.cfg.buffer_config

        # --- Storage ---
        # - obs[t] — observation at time t
        self.obs = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, state.flat_obs_dim, device=self.device)

        self.actions = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, 1, device=self.device)

        # - rewards[t] — reward for transition t → t+1
        self.rewards = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, device=self.device)

        self.values = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, device=self.device)

        self.logprobs = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, device=self.device)

        self.terminated = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, dtype=torch.bool, device=self.device)

        self.truncated = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, dtype=torch.bool, device=self.device)

        # Hidden states at *start* of each timestep
        # - hxs[t], cxs[t] — LSTM state at start of timestep t
        self.hxs = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, self.cfg.lstm_hidden_size, device=self.device)
        self.cxs = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, self.cfg.lstm_hidden_size, device=self.device)

        # - returns[t] — GAE‑computed value target
        self.returns = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, device=self.device)

        # - advantages[t] — GAE advantage
        self.advantages = torch.zeros(self.cfg.rollout_steps, self.cfg.num_envs, device=self.device)

        self.masks: torch.Tensor | None = None

        self.reset()

    # ---------------------------------------------------------
    # Add rollout step
    # ---------------------------------------------------------
    def add(self, step: RolloutStep):
        # --- Pointer safety ---
        t = self.step

        """
        Assertion that protects the write path
        ----------------------------------------------------
        - Prevents writing past the end of the buffer
        - Prevents silent corruption of rollout data
        - Prevents TBPTT slicing errors
        - Prevents hidden‑state alignment failures
        - Makes your GPU tests meaningful
        """
        assert t < self.cfg.rollout_steps, f"RolloutBuffer overflow: step={t}, max={self.cfg.rollout_steps}"

        # --- Shape checks ---
        assert step.obs.shape == (self.cfg.num_envs, self.obs.size(-1)), f"Obs shape mismatch: {step.obs.shape}"

        assert step.hxs.shape == (self.cfg.num_envs, self.cfg.lstm_hidden_size)

        assert step.cxs.shape == (self.cfg.num_envs, self.cfg.lstm_hidden_size)

        # --- Store rollout data ---
        self.obs[t].copy_(step.obs)  # step.obs is already flat

        # Actions must be (B,1)
        if step.actions.dim() == 1:
            self.actions[t].copy_(step.actions.unsqueeze(-1))
        elif step.actions.dim() == 2 and step.actions.size(-1) == 1:
            self.actions[t].copy_(step.actions)
        else:
            raise RuntimeError(f"Invalid action shape: {step.actions.shape}")

        self.rewards[t].copy_(step.rewards)
        self.values[t].copy_(step.values)
        self.logprobs[t].copy_(step.logprobs)
        self.terminated[t].copy_(step.terminated)
        self.truncated[t].copy_(step.truncated)

        """
        Hidden state at *start* of timestep t
        ------------------------------------------------------------
        LSTM State‑Flow Invariant: The buffer must store the PRE‑STEP LSTM state (h_t, c_t) that was used to produce
        action[t]. This state is the anchor for:

        • Deterministic state‑flow validation
        • Correct PPO recurrent evaluation (evaluate_actions_sequence)
        • TBPTT chunking (each chunk begins at the true h_t, c_t)
        • Accurate LSTM diagnostics (drift, saturation, entropy)

        Storing the POST‑STEP state (h_{t+1}, c_{t+1}) would silently break the entire recurrent pipeline:

        • validate_lstm_state_flow() fails
        • TBPTT unrolls from the wrong state
        • PPO logprobs/values become misaligned
        • Diagnostics become meaningless

        => Never modify this without re‑running all LSTM validation tests.
        """
        self.hxs[t].copy_(step.hxs)
        self.cxs[t].copy_(step.cxs)

        self.step += 1

    # ---------------------------------------------------------
    # Store final LSTM states for next rollout
    # ---------------------------------------------------------
    def store_last_lstm_states(self, last_policy_output):
        """
        Stores the PRE‑STEP LSTM state used to produce the *next* action.

        LSTM Rollout Boundary Invariant:
        --------------------------------
        At the end of a rollout, PPO must carry forward the LSTM state (h_T, c_T) that existed *before* the final
        env.step(). This state becomes the initial hidden state for the next rollout.

        Critical requirements:
        ---------------------
        • new_hxs/new_cxs from the policy are shaped (T, B, H) or (B, H) depending on sequence length; we must extract
          the final timestep.
        • The stored state must be 2‑D (B, H) to match rollout initialization.
        • The state must be detached to prevent gradients from spanning across rollout boundaries.

        Why this invariant matters:
        ---------------------------
        • Ensures deterministic state‑flow across rollouts
        • Ensures evaluate_actions_sequence() reproduces rollout‑time behavior
        • Ensures TBPTT chunking begins from the correct h_0, c_0
        • Ensures LSTM diagnostics (drift, saturation, entropy) remain aligned
        • Prevents accidental backprop through multiple rollouts

        If this invariant is violated:
        ------------------------------
        • validate_lstm_state_flow() will fail
        • PPO will compute logprobs/values from misaligned states
        • TBPTT will unroll from the wrong initial state
        • Diagnostics become meaningless
        • Training becomes nondeterministic

        => Never modify this logic without re‑running all LSTM validation tests.
        """
        # Typically new_hxs/new_cxs are (T, B, H) or (T, H) when B=1
        hxs = last_policy_output.new_hxs[-1]
        cxs = last_policy_output.new_cxs[-1]

        # 🔒 Invariant: always store as (B, H)
        if hxs.dim() == 1:
            hxs = hxs.unsqueeze(0)
            cxs = cxs.unsqueeze(0)

        self.last_hxs = hxs.detach()
        self.last_cxs = cxs.detach()

    # ---------------------------------------------------------
    # GAE-Lambda
    # ---------------------------------------------------------
    def compute_returns_and_advantages(self, last_value):
        """
        Reward & Return normalization operate at  different stages and
        solve different stability problems.

        Reward normalization
        ---------------------
        stabilizes the advantage calculation because GAE uses rewards
        directly

        Return normalization
        --------------------
        stabilizes the critic’s regression target because the value loss
        uses returns directly.

        ####################
        Reward Normalization
        ####################

        What it normalizes:
        ------------------
            The raw rewards coming from the environment (after shaping).

        Where it happens:
        -----------------
            Inside the rollout buffer, before computing GAE.

        Why it exists:
        --------------
            Reward normalization stabilizes the critic’s TD error by ensuring
            that the reward scale is consistent across rollouts.

        Why it matters:
        --------------
        • 	Shaping increases reward variance
        • 	Sparse rewards cause huge variance
        • 	Dense rewards can explode the critic
        • 	PPO’s GAE is sensitive to reward scale
        • 	LSTMs amplify variance over long horizons

        Effect:
        ------
        Reward normalization makes the advantage signal smoother, which makes
        the critic learn faster and prevents value explosion.

        Analogy:
        -------
        Reward normalization is like normalizing your input features before
        training a neural network.
        """
        r = self.rewards
        self.rewards = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)

        last_gae = torch.zeros(self.cfg.num_envs, device=self.device)

        for t in reversed(range(self.cfg.rollout_steps)):
            true_terminal = self.terminated[t]

            # truncated episodes bootstrap; terminated do not
            bootstrap = ~true_terminal

            next_value = last_value if t == self.cfg.rollout_steps - 1 else self.values[t + 1]

            delta = self.rewards[t] + self.cfg.gamma * next_value * bootstrap - self.values[t]

            last_gae = delta + self.cfg.gamma * self.cfg.lam * last_gae * bootstrap

            self.advantages[t] = last_gae

        self.returns = self.values + self.advantages

        """
        #############################################
        Return Normalization (Value-target Whitening)
        #############################################

        What it normalizes:
        ------------------
        The discounted returns (value targets) after GAE.

        Where it happens:
        ----------------
        Inside the rollout buffer, after computing returns.

        Why it exists:
        --------------
        Return normalization stabilizes the value function regression
        by ensuring the critic always predicts targets with roughly zero mean
        and unit variance.

        Why it matters:
        --------------
        • 	PPO’s value loss is a regression problem
        • 	If returns are small early and large later, the critic becomes
            unstable
        • 	In CartPole‑PO, returns are extremely low early on
        • 	Without normalization, the critic collapses or learns extremely
        slowly

        Effect:
        ------
        Return normalization makes the critic’s regression target stable,
        which improves explained variance and prevents critic drift.

        Analogy:
        -------
        Return normalization is like whitening your labels in a regression
        problem.
        """
        ret = self.returns
        self.returns = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-8)

    # ---------------------------------------------------------
    # Yield minibatches of full sequences (T, B, ...)
    # ---------------------------------------------------------
    def get_recurrent_minibatches(self):
        """
        Yield full‑sequence recurrent minibatches in time‑major format.

        get_recurrent_minibatches Invariants:
        -------------------------------------
        This function defines how PPO samples (T, B, ...) sequences from the rollout buffer. It must preserve the
        temporal structure of the rollout and maintain alignment across all tensors used by the recurrent policy.

        Critical invariants:

        1. Time‑major slicing
        ---------------------
        All tensors must be sliced as (T, B, ...) so that PPO, TBPTT, and evaluate_actions_sequence() operate on
        identically aligned data.

        2. Environment‑major shuffling only
        -----------------------------------
        We shuffle environments (the B dimension) but never shuffle or break the temporal dimension T. Each minibatch
        contains full sequences for a subset of environments.

        3. PRE‑STEP hidden state alignment
        ----------------------------------
        The hxs/cxs tensors sliced here must correspond to the PRE‑STEP states stored during rollout. This ensures that
        TBPTT chunks begin from the correct initial hidden state.

        4. next_obs / next_rewards alignment
        ------------------------------------
        next_obs[t] = obs[t+1] and next_rewards[t] = rewards[t] must be constructed here so that auxiliary prediction
        targets remain aligned with PPO’s time‑major rollout structure.

        5. Mask correctness
        -------------------
        terminated/truncated flags must be sliced in (T, B) format so that TBPTT and PPO losses correctly ignore
        invalid timesteps.

        6. No gradient detachment
        -------------------------
        All tensors returned here must retain gradients through the policy evaluation path. Only hidden states are
        detached later, never here.

        If any of these invariants are violated, PPO training becomes nondeterministic, TBPTT breaks, auxiliary losses
        misalign, and LSTM state‑flow validation fails.

        => Never modify this function without re‑running all recurrent PPO tests.
        """
        env_indices = torch.randperm(self.cfg.num_envs, device=self.device)

        for start in range(0, self.cfg.num_envs, self.cfg.mini_batch_envs):
            idx = env_indices[start : start + self.cfg.mini_batch_envs]

            # Slice rollout data
            obs = self.obs[:, idx]  # (T, B, obs_dim)
            rewards = self.rewards[:, idx]  # (T, B)

            """
            Auxiliary Prediction Invariants (next_obs / next_rewards)
            ---------------------------------------------------------
            Auxiliary tasks predict the next observation and next reward for each timestep. These targets must be
            constructed in a way that preserves the temporal structure of the rollout and aligns with PPO’s time‑major
            format.

            Critical invariants:

            1. next_obs[t] = obs[t+1]
            -------------------------
            The next observation is the environment state *after* taking action[t]. Because the rollout buffer stores
            observations in time‑major format (T, B, ...), next_obs is obtained by shifting obs by one timestep. The
            final timestep is padded (and masked out) because obs[T+1] does not exist within the rollout.

            2. next_rewards[t] = rewards[t]
            -------------------------------
            In Gym/PopGym environments, reward[t] is already the reward for the transition t → t+1. There is no
            separate “next reward” field. Using rewards[t] preserves PPO’s GAE semantics and ensures that auxiliary
            reward prediction aligns with the true transition function.

            3. Sequence‑level alignment
            ---------------------------
            next_obs and next_rewards must have shape (T, B, ...) so they remain aligned with obs, actions, values,
            returns, advantages, and masks. This avoids any special‑case handling in TBPTT or the trainer.

            4. Mask correctness
            -------------------
            The final timestep’s next_obs is padded but always masked out by the (terminated | truncated) mask.
            Auxiliary losses must never propagate gradients through invalid timesteps.

            5. No gradient detachment
            -------------------------
            next_obs and next_rewards are treated as supervised targets; the predictions retain gradients, but the
            targets do not. Detaching targets here ensures PPO and auxiliary losses remain independent.

            If any of these invariants are violated, auxiliary losses become misaligned, TBPTT breaks, and PPO’s
            recurrent evaluation no longer matches rollout‑time behavior.

            => Never modify this logic without re‑running all recurrent PPO tests.
            """
            next_obs = obs[1:]  # (T-1, B, obs_dim)

            # Pad final timestep (masked out anyway)
            pad = torch.zeros_like(next_obs[0:1])
            next_obs = torch.cat([next_obs, pad], dim=0)

            # the reward at time t is already the reward for the transition t → t+1.
            next_rewards = rewards  # already aligned

            yield RecurrentBatch(
                obs=obs,
                next_obs=next_obs,
                next_rewards=next_rewards,
                actions=self.actions[:, idx],
                values=self.values[:, idx],
                logprobs=self.logprobs[:, idx],
                returns=self.returns[:, idx],
                advantages=self.advantages[:, idx],
                hxs=self.hxs[:, idx],
                cxs=self.cxs[:, idx],
                terminated=self.terminated[:, idx],
                truncated=self.truncated[:, idx],
            )

    # ---------------------------------------------------------
    # LSTM state handoff to env wrapper
    # ---------------------------------------------------------
    def get_last_lstm_states(self):
        hxs = self.last_hxs
        cxs = self.last_cxs

        assert isinstance(hxs, torch.Tensor)
        assert isinstance(cxs, torch.Tensor)

        return LSTMStates(hxs=hxs, cxs=cxs)

    # ---------------------------------------------------------
    # Reset buffer
    # ---------------------------------------------------------
    def reset(self):
        """
        Reset attrributes to ensure

        - rollout correctness
        - TBPTT correctness
        - hidden‑state alignment
        - mask correctness
        - replay determinism
        - drift/saturation/entropy correctness
        """

        # Pointer
        self.step = 0

        # Last-step LSTM states (for next rollout)
        self.last_hxs = None
        self.last_cxs = None

        # Clear rollout storage
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.logprobs.zero_()

        # Episode termination flags
        self.terminated.zero_()
        self.truncated.zero_()

        """
        - .mask is great for quick access
        - .masks is needed for diagnostics, TBPTT, and replay
        - Trainer code stays unchanged
        - No performance penalty (you compute mask once per rollout)
        """
        self.masks = self.mask  # property → tensor

        # Hidden states at start of each timestep
        self.hxs.zero_()
        self.cxs.zero_()

        # Returns and advantages
        self.returns.zero_()
        self.advantages.zero_()

    @property
    def mask(self):
        """
        Mask of valid timesteps: 1 for alive, 0 for terminated/truncated.
        Shape: (T, B)
        """
        return 1.0 - (self.terminated | self.truncated).float()

    # ---------------------------------------------------------
    # Optional safety check
    # ---------------------------------------------------------
    def finalize(self):
        assert self.step == self.cfg.rollout_steps, f"Rollout incomplete: {self.step}/{self.cfg.rollout_steps}"
