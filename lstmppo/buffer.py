import torch
from .types import Config, RolloutStep, RecurrentBatch, LSTMStates


class RecurrentRolloutBuffer:

    def __init__(self,
                 cfg: Config,
                 device):

        self.device = device

        self.cfg = cfg.buffer_config

        # --- Storage ---
        self.obs = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            cfg.env.flat_obs_dim,
            device=self.device
        )

        self.actions = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            1,
            device=self.device
        )

        self.rewards = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            device=self.device
        )

        self.values = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            device=self.device
        )

        self.logprobs = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            device=self.device
        )

        self.terminated = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            dtype=torch.bool,
            device=self.device
        )

        self.truncated = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            dtype=torch.bool,
            device=self.device
        )

        # Hidden states at *start* of each timestep
        self.hxs = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            self.cfg.lstm_hidden_size,
            device=self.device
        )

        self.cxs = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            self.cfg.lstm_hidden_size,
            device=self.device
        )

        self.returns = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            device=self.device
        )

        self.advantages = torch.zeros(
            self.cfg.rollout_steps,
            self.cfg.num_envs,
            device=self.device
        )

        self.reset()

    # ---------------------------------------------------------
    # Add rollout step
    # ---------------------------------------------------------
    def add(self,
            step: RolloutStep):

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
        assert t < self.cfg.rollout_steps, \
            f"RolloutBuffer overflow: step={t}, max={self.cfg.rollout_steps}"

        # --- Shape checks ---
        assert step.obs.shape == (self.cfg.num_envs, self.obs.size(-1)), \
            f"Obs shape mismatch: {step.obs.shape}"

        assert step.hxs.shape == (self.cfg.num_envs,
                                  self.cfg.lstm_hidden_size)
        
        assert step.cxs.shape == (self.cfg.num_envs,
                                  self.cfg.lstm_hidden_size)

        # --- Store rollout data ---
        self.obs[t].copy_(step.obs)   # step.obs is already flat

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

        # Hidden state at *start* of timestep t
        self.hxs[t].copy_(step.hxs)
        self.cxs[t].copy_(step.cxs)

        self.step += 1

    # ---------------------------------------------------------
    # Store final LSTM states for next rollout
    # ---------------------------------------------------------
    def store_last_lstm_states(self,
                               last_policy_output):

        # last timestep hidden state (B, H)
        self.last_hxs = last_policy_output.new_hxs[-1].detach()
        self.last_cxs = last_policy_output.new_cxs[-1].detach()

    # ---------------------------------------------------------
    # GAE-Lambda
    # ---------------------------------------------------------
    def compute_returns_and_advantages(self,
                                       last_value):
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

        last_gae = torch.zeros(self.cfg.num_envs,
                               device=self.device)

        for t in reversed(range(self.cfg.rollout_steps)):

            true_terminal = self.terminated[t]

            # truncated episodes bootstrap; terminated do not
            bootstrap = ~true_terminal

            next_value = (
                last_value if t == self.cfg.rollout_steps - 1
                else self.values[t + 1]
            )

            delta = (
                self.rewards[t]
                + self.cfg.gamma * next_value * bootstrap
                - self.values[t]
            )

            last_gae = (
                delta + self.cfg.gamma
                * self.cfg.lam * last_gae * bootstrap
            )

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

        env_indices = torch.randperm(self.cfg.num_envs,
                                     device=self.device)

        for start in range(0,
                           self.cfg.num_envs,
                           self.cfg.mini_batch_envs):

            idx = env_indices[start:start + self.cfg.mini_batch_envs]

            yield RecurrentBatch(
                obs=self.obs[:, idx],
                actions=self.actions[:, idx],
                values=self.values[:, idx],
                logprobs=self.logprobs[:, idx],
                returns=self.returns[:, idx],
                advantages=self.advantages[:, idx],
                hxs=self.hxs[:, idx],
                cxs=self.cxs[:, idx],
                terminated=self.terminated[:, idx],
                truncated=self.truncated[:, idx]
            )

    # ---------------------------------------------------------
    # LSTM state handoff to env wrapper
    # ---------------------------------------------------------
    def get_last_lstm_states(self):

        return LSTMStates(
            hxs=self.last_hxs,
            cxs=self.last_cxs
        )

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

        # Hidden states at start of each timestep
        self.hxs.zero_()
        self.cxs.zero_()

        # Returns and advantages
        self.returns.zero_()
        self.advantages.zero_()

        """
        Ensure

        - all environments are “alive” at the start of a rollout
        - GAE bootstrapping works
        - PPO loss masking works
        - drift/saturation/entropy masking works
        - replay determinism works

        because mask correctness is one of the most important invariants in
        an LSTM-PPO pipeline.
        """
        self.mask = torch.ones_like(self.rewards)

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
        assert self.step == self.cfg.rollout_steps, \
            f"Rollout incomplete: {self.step}/{self.cfg.rollout_steps}"