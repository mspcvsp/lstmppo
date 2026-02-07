import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .obs_encoder import build_obs_encoder
from .trainer_state import TrainerState
from .types import LSTMCoreOutput, LSTMGates, PolicyEvalInput, PolicyEvalOutput, PolicyInput, PolicyOutput


class GateLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropconnect_p, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropconnect_p = dropconnect_p
        self.debug_mode = kwargs.get("debug_mode", False)

        # Input weights
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))

        # Recurrent weights (raw)
        self.weight_hh_raw = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        # Biases
        if kwargs.get("bias", True):
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        # Initialize everything
        self.reset_parameters()

        self.register_parameter("weight_hh_raw", self.weight_hh_raw)

        # Temporary tensor, NOT a parameter
        self.weight_hh = None

    def reset_parameters(self):
        if self.weight_ih.numel() > 0:
            nn.init.xavier_uniform_(self.weight_ih)

        if self.weight_hh_raw.numel() > 0:
            nn.init.orthogonal_(self.weight_hh_raw)

        if self.bias_ih is not None and self.bias_ih.numel() > 0:
            nn.init.zeros_(self.bias_ih)

            # Forget gate bias trick
            H = self.hidden_size
            self.bias_ih.data[H : 2 * H] = 1.0

        if self.bias_hh is not None and self.bias_hh.numel() > 0:
            nn.init.zeros_(self.bias_hh)

    def _apply_dropconnect(self):
        if self.training and not self.debug_mode:
            mask = torch.ones_like(self.weight_hh_raw)

            mask = F.dropout(mask, p=self.dropconnect_p, training=True)
            self.weight_hh = self.weight_hh_raw * mask

        else:
            self.weight_hh = self.weight_hh_raw

    def forward(self, x, hx):
        h, c = hx  # each (B, H)

        self._apply_dropconnect()

        assert isinstance(x, torch.Tensor)
        assert isinstance(h, torch.Tensor)
        assert isinstance(self.weight_ih, torch.Tensor)
        assert isinstance(self.weight_hh, torch.Tensor)
        assert isinstance(self.bias_ih, torch.Tensor)
        assert isinstance(self.bias_hh, torch.Tensor)

        gates = x @ self.weight_ih.t() + h @ self.weight_hh.t() + self.bias_ih + self.bias_hh

        H = self.hidden_size
        i = torch.sigmoid(gates[:, :H])
        f = torch.sigmoid(gates[:, H : 2 * H])
        g = torch.tanh(gates[:, 2 * H : 3 * H])
        o = torch.sigmoid(gates[:, 3 * H : 4 * H])

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)

        return new_h, new_c, (i, f, g, o)


class LSTMCore(nn.Module):
    def __init__(self, cell: GateLSTMCell):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size

    def initial_state(self, batch_size: int, device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c

    def forward(self, x, state):
        # MUST return (h, c, gates)
        return self.cell(x, state)


class LSTMPPOPolicy(nn.Module):
    def __init__(self, state: TrainerState):
        super().__init__()
        assert state.env_info is not None

        self.ar_coef = state.cfg.lstm.lstm_ar_coef
        self.tar_coef = state.cfg.lstm.lstm_tar_coef

        # If obs_space is None, rely on flat_obs_dim (tests do this intentionally)
        if state.env_info.obs_space is None:
            self.obs_dim = 0
        else:
            self.obs_dim = state.env_info.flat_obs_dim

        obs_space = state.env_info.obs_space
        self.obs_encoder = build_obs_encoder(obs_space, self.obs_dim)

        # --- SiLU encoder ---
        if state.env_info.flat_obs_dim == 0:
            """
            When obs_dim == 0, the environment provides no observation
            features. But the LSTM still needs an input vector of size
            enc_hidden_size. So the correct behavior is:

            - Treat the encoder as a constant zero‑feature generator
            - Let the LSTM operate purely on its recurrent state
            - Preserve shape invariants everywhere

            This is exactly how RL libraries handle “no observation” cases.
            """
            self.encoder = ZeroFeatureEncoder(state.cfg.lstm.enc_hidden_size)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(state.env_info.flat_obs_dim, state.cfg.lstm.enc_hidden_size),
                nn.SiLU(),
                nn.Linear(state.cfg.lstm.enc_hidden_size, state.cfg.lstm.enc_hidden_size),
                nn.SiLU(),
            )

        if isinstance(self.encoder, nn.Sequential):
            for m in self.encoder:
                if isinstance(m, nn.Linear):
                    if m.weight.numel() > 0:
                        nn.init.xavier_uniform_(m.weight)

                    if m.bias.numel() > 0:
                        nn.init.zeros_(m.bias)

        # --- LN-LSTM with DropConnect ---
        self.lstm_cell = GateLSTMCell(
            input_size=state.cfg.lstm.enc_hidden_size,
            hidden_size=state.cfg.lstm.lstm_hidden_size,
            dropconnect_p=state.cfg.lstm.dropconnect_p,
            debug_mode=state.cfg.trainer.debug_mode,
        )

        # Wrap it so diagnostics can call policy.lstm.initial_state(...)
        self.lstm = LSTMCore(self.lstm_cell)

        self.ln = nn.LayerNorm(state.cfg.lstm.lstm_hidden_size)

        # --- Heads ---
        if state.env_info.action_dim == 0:
            self.actor = nn.Identity()
        else:
            self.actor = nn.Linear(state.cfg.lstm.lstm_hidden_size, state.env_info.action_dim)

        self.critic = nn.Linear(state.cfg.lstm.lstm_hidden_size, 1)

        self.obs_pred_head = nn.Linear(state.cfg.lstm.lstm_hidden_size, self.obs_dim)
        self.rew_pred_head = nn.Linear(state.cfg.lstm.lstm_hidden_size, 1)

        if isinstance(self.actor, nn.Linear):
            if self.actor.weight.numel() > 0:
                nn.init.xavier_uniform_(self.actor.weight)

            if self.actor.bias.numel() > 0:
                nn.init.zeros_(self.actor.bias)

        if self.critic.weight.numel() > 0:
            nn.init.xavier_uniform_(self.critic.weight)

        if self.critic.bias.numel() > 0:
            nn.init.zeros_(self.critic.bias)

        if self.obs_pred_head.weight.numel() > 0:
            nn.init.xavier_uniform_(self.obs_pred_head.weight)

        if self.obs_pred_head.bias.numel() > 0:
            nn.init.zeros_(self.obs_pred_head.bias)

        if self.rew_pred_head.weight.numel() > 0:
            nn.init.xavier_uniform_(self.rew_pred_head.weight)

        if self.rew_pred_head.bias.numel() > 0:
            nn.init.zeros_(self.rew_pred_head.bias)

    def initial_state(self, batch_size: int, device):
        return self.lstm.initial_state(batch_size, device)

    # ---------------------------------------------------------
    # Core forward pass (returns activations + AR/TAR)
    # ---------------------------------------------------------
    def _forward_core(self, x, hxs, cxs):
        """
        Core LSTM unroll over a full (B, T, ...) sequence.

        _forward_core Invariants:
        -------------------------
        This function defines the authoritative recurrent computation used by both rollout-time action selection and
        PPO training-time evaluation. Any change here must preserve the following invariants:

        1. PRE‑STEP hidden state semantics
        ----------------------------------
        The input hxs/cxs must represent the PRE‑STEP state for timestep 0. This ensures that rollout-time behavior and
        training-time evaluation produce identical logprobs, values, and hidden-state transitions.

        2. Full LSTM unroll (no truncation)
        -----------------------------------
        The LSTM must be unrolled across the entire sequence so that:
            • h_{t+1}, c_{t+1} reflect the true temporal dynamics
            • gate activations (i, f, g, o) are aligned with timestep t
            • evaluate_actions_sequence() can reproduce rollout behavior
            • TBPTT chunking remains correct

        3. Batch‑first internal format
        ------------------------------
        Internally, the model operates in batch‑first format (B, T, H), even though the rollout buffer stores
        everything in time‑major format (T, B, ...). All outputs must be transposed back to time‑major before
        returning.

        4. No gradient detachment inside the unroll
        ---------------------------------------
        PPO requires gradients through:
            • logits → logprobs → policy loss
            • values → value loss
            • entropy → entropy bonus

        Only the returned hidden states (new_hxs/new_cxs) and diagnostics may be detached to prevent gradients across
        rollout boundaries.

        5. Diagnostic alignment
        -----------------------
        Gate activations, cell states, hidden states, and any auxiliary metrics must be recorded in (T, B, ...) format
        so that drift, saturation, and entropy diagnostics remain temporally aligned with rollout data.

        If any of these invariants are violated, PPO training becomes nondeterministic, TBPTT breaks, and LSTM
        diagnostics become invalid.

        => Never modify this function without re-running all LSTM state-flow and TBPTT validation tests.
        """
        obs_flat = self.obs_encoder(x)  # x may be dict/tuple/box

        if obs_flat.dim() == 2:
            obs_flat = obs_flat.unsqueeze(1)  # (B,1,F)

        # (B,T,H) for both normal and zero encoder
        enc = self.encoder(obs_flat)

        # LSTM expects (B, T, F) with batch_first=True
        B, T, F = enc.shape
        h = hxs
        c = cxs

        outputs = []
        gate_list = []

        """
        The full [T,B,H] sequences need to be recorded in order to compute
        saturation metrics over the entire sequence
        """
        c_list = []
        h_list = []

        for t in range(T):
            h, c, gates = self.lstm(enc[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
            gate_list.append(gates)

            # store per‑timestep hidden and cell states [B, H]
            h_list.append(h)
            c_list.append(c)

        out = torch.cat(outputs, dim=1)  # (B, T, H)
        pred_obs = self.obs_pred_head(out)  # (B, T, obs_dim)
        pred_rew = self.rew_pred_head(out)  # (B, T, 1)

        i_gates, f_gates, g_gates, o_gates = zip(*gate_list)

        # Stack into (B, T, H)
        # Detaching inside the loop creates many small detached tensors
        i_gates = torch.stack(i_gates, dim=1).detach()
        f_gates = torch.stack(f_gates, dim=1).detach()
        g_gates = torch.stack(g_gates, dim=1).detach()
        o_gates = torch.stack(o_gates, dim=1).detach()
        h_gates = torch.stack(h_list, dim=1).detach()
        c_gates = torch.stack(c_list, dim=1).detach()

        # --- Activation Regularization (AR) ---
        ar_loss = (out.pow(2).mean()) * self.ar_coef

        # --- Temporal Activation Regularization (TAR) ---
        if out.size(1) > 1:
            tar_loss = (out[:, 1:, :] - out[:, :-1, :]).pow(2).mean() * self.tar_coef
        else:
            tar_loss = torch.tensor(0.0, device=out.device)

        return LSTMCoreOutput(
            out=out,
            pred_obs=pred_obs,
            pred_raw=pred_rew,
            h=h,
            c=c,
            ar_loss=ar_loss,
            tar_loss=tar_loss,
            gates=LSTMGates(
                i_gates=i_gates,
                f_gates=f_gates,
                g_gates=g_gates,
                o_gates=o_gates,
                c_gates=c_gates,
                h_gates=h_gates,
            ),
        )

    # ---------------------------------------------------------
    # Dataclass-based forward
    # ---------------------------------------------------------
    def forward(self, inp: PolicyInput) -> PolicyOutput:
        """
        inp.obs: (B, obs_dim) OR (B, T, obs_dim)
        inp.hxs, inp.cxs: (B, H)
        """
        core_out = self._forward_core(inp.obs, inp.hxs, inp.cxs)

        # core_out is ALWAYS (B, T, H)
        B, T, H = core_out.out.shape
        flat = core_out.out.reshape(B * T, H)

        if isinstance(self.actor, nn.Identity):
            logits = torch.zeros(B, T, 0, device=flat.device)
        else:
            logits = self.actor(flat).view(B, T, -1)  # (B, T, A)

        values = self.critic(flat).view(B, T)  # (B, T)

        # If single-step, squeeze back to (B, A) and (B,)
        if T == 1:
            logits = logits[:, 0, :]
            values = values[:, 0]

        # logits: (B, T, A) for sequences, (B, A) for single-step
        # values: (B, T) or (B,)
        return PolicyOutput(
            logits=logits,
            values=values,
            pred_obs=core_out.pred_obs,
            pred_raw=core_out.pred_raw,
            new_hxs=core_out.h,
            new_cxs=core_out.c,
            ar_loss=core_out.ar_loss,
            tar_loss=core_out.tar_loss,
            gates=core_out.gates,
        )

    def cell(self, x, state):
        # x: (B, H)
        # state: (h, c)
        h, c = state
        gates = self.lstm_cell(x, (h, c))  # or however your cell is implemented
        return gates  # must return (h_new, c_new)

    def forward_step(self, obs, h, c):
        """
        Single‑step recurrent policy evaluation used during rollout.

        forward_step Invariants:
        ------------------------
        This function must reproduce the exact PRE‑STEP → POST‑STEP LSTM transition that occurs inside `_forward_core`,
        but for a single timestep (B, ...) instead of a full sequence (B, T, ...).

        Critical invariants:

        1. PRE‑STEP hidden state semantics
        ----------------------------------
        The input hxs/cxs must be the PRE‑STEP state (h_t, c_t) used to produce action[t]. This ensures that
        rollout‑time behavior matches the sequence unroll performed during PPO training.

        2. Exact LSTM transition
        ------------------------
        The update (h_{t+1}, c_{t+1}) computed here must be bit‑identical to the transition computed inside
        `_forward_core` for the same input. Any drift breaks:

            • state‑flow validation
            • evaluate_actions_sequence() alignment
            • TBPTT correctness

        3. No gradient detachment inside the step
        -----------------------------------------
        The logits, values, and entropy must retain gradients so PPO can compute correct losses during training. Only
        the returned hidden states (new_hxs/new_cxs) may be detached to prevent gradients from spanning across rollout
        boundaries.

        4. Batch‑first consistency
        --------------------------
        Even though rollout uses single steps, the internal LSTM expects batch‑first format. Shapes must match the
        conventions used in `_forward_core` so that both paths remain interchangeable.

        5. Diagnostic alignment
        -----------------------
        Any gate activations or auxiliary metrics produced here must match the shapes and semantics of those produced
        in `_forward_core`, ensuring that drift/saturation/entropy diagnostics remain valid.

        If any of these invariants are violated, rollout‑time behavior will diverge from training‑time evaluation,
        breaking PPO, TBPTT, and all LSTM state‑flow diagnostics.

        => Never modify this function without re‑running all LSTM validation tests.
        """
        enc = self.encoder(self.obs_encoder(obs))  # (B, H)
        h, c, gates = self.lstm(enc, (h, c))
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value, h, c, gates

    def act(self, policy_input: PolicyInput):
        policy_output = self.forward(policy_input)

        dist = Categorical(logits=policy_output.logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions, logprobs, policy_output

    def evaluate_actions_sequence(self, inp: PolicyEvalInput) -> PolicyEvalOutput:
        """
        Fully sequence-aware PPO evaluation.

        inp.obs:     (T, B, obs_dim)
        inp.hxs:     (B, H)
        inp.cxs:     (B, H)
        inp.actions: (T, B) or (T, B, 1)

        ---------------------------------------------------------------------------------
        Sequence‑aware PPO evaluation (training‑time only)
        ---------------------------------------------------------------------------------

        LSTM Evaluation Invariant:
        --------------------------
        This function must reproduce the exact recurrent state‑flow and logprob/value computation that occurred during
        rollout, but over entire (T, B, ...) sequences instead of single steps.

        Critical invariants enforced here:

        1. Batch‑first → Time‑major alignment

            The policy forward pass operates in batch‑first format (B, T, H), while the rollout buffer stores
            everything in time‑major format (T, B, ...). All tensors (logits, values, gates, hxs, cxs) must be
            transposed back to (T, B, ...) to remain aligned with PPO’s minibatch slicing and masking.

        2. PRE‑STEP hidden state consistency

            The initial hxs/cxs passed in must be the PRE‑STEP states stored during rollout. This guarantees that
            evaluate_actions_sequence() produces logprobs/values identical to rollout‑time behavior.

        3. Full LSTM unroll (no truncation)

            Unlike act(), which performs a single‑step transition, this method must unroll the LSTM over the entire
            sequence so that:

            • logprobs[t] match the correct logits[t]
            • values[t] match the correct critic outputs
            • gate diagnostics (i,f,g,o,h,c) reflect the true temporal
                structure of the rollout

        4. No detaching of logprobs/values - PPO requires gradients through:

            • logprobs → policy gradient
            • values   → value loss
            • entropy  → entropy bonus

            Only hidden states (new_hxs/new_cxs) and gate diagnostics are detached to prevent gradients across rollout
            boundaries. If any of these invariants are violated, PPO training becomes nondeterministic, TBPTT breaks,
            and LSTM diagnostics become misaligned or meaningless.
        """
        T, B = inp.obs.shape[0], inp.obs.shape[1]

        # (T, B, F) -> (B, T, F) for batch_first LSTM
        obs_bt = inp.obs.transpose(0, 1)  # (B, T, obs_dim)

        policy_input = PolicyInput(
            obs=obs_bt,  # (B, T, obs_dim)
            hxs=inp.hxs,  # (B, H)
            cxs=inp.cxs,  # (B, H)
        )

        # Forward through encoder + LSTM + heads
        policy_output = self.forward(policy_input)
        # policy_output.logits: (B, T, A)
        # policy_output.values: (B, T)

        """
        Why pred_obs and pred_rew are returned here:
        -------------------------------------------
        Auxiliary prediction (next‑observation and next‑reward prediction) is a training‑time feature. PPO computes
        these auxiliary losses over full (T, B, ...) sequences, so evaluate_actions_sequence() must expose the
        model’s predicted next‑state and next‑reward tensors in time‑major format.

        During rollout, evaluate_actions() is single‑step and does not use auxiliary predictions, so pred_obs and
        pred_rew are omitted there to avoid unnecessary computation and keep the rollout path lightweight.

        In short:

        - evaluate_actions() → rollout‑time, single‑step, no auxiliary predictions
        - evaluate_actions_sequence() → training‑time, full‑sequence, returns

        => pred_obs and pred_rew for auxiliary losses

        This separation keeps the rollout path fast and the training path fully supervised without mixing concerns.
        """
        pred_obs = policy_output.pred_obs.transpose(0, 1)  # (T, B, obs_dim)
        pred_raw = policy_output.pred_raw.transpose(0, 1)  # (T, B, 1)

        # Back to (T, B, ...)
        logits = policy_output.logits.transpose(0, 1)  # (T, B, A)
        values = policy_output.values.transpose(0, 1)  # (T, B)

        # Actions: ensure shape (T, B)
        if inp.actions.dim() == 3 and inp.actions.size(-1) == 1:
            actions = inp.actions.squeeze(-1)  # (T, B)
        else:
            actions = inp.actions  # (T, B)

        dist = self._dist_from_logits(logits)

        assert logits.dim() == 3, f"logits must be (T,B,A), got {logits.shape}"

        assert actions.shape == (T, B), f"actions must be (T,B), got {actions.shape}"

        # Correct shape: (T, B)
        logprobs = dist.log_prob(actions)  # (T, B)
        entropy = dist.entropy()  # (T, B)

        """
        _forward_core returns batch‑first tensors shaped (B, T, H).

        However, the PPO training pipeline stores and processes all rollout
        data in time‑major format (T, B, ...). This includes:

        - logprobs
        - entropy
        - values
        - advantages
        - masks
        - gate drift
        - gate saturation
        - gate entropy

        To keep all diagnostics and losses aligned with rollout storage and
        minibatch slicing, gate tensors must be transposed to (T, B, H).
        """
        gates = policy_output.gates.detached.transposed()

        """
        hxs and cxs are recurrent state outputs from the LSTM that are fed
        into the next rollout step. Detach to prevent gradients flowing
        across rollout boundaries. PPO treats each rollout as a truncated
        BPTT segment
        """
        # --- Make new_hxs/new_cxs time-major (T, B, H) robustly ---
        new_hxs = policy_output.new_hxs.detach()
        new_cxs = policy_output.new_cxs.detach()

        if new_hxs.dim() == 2:
            # (B, H) → (T=1, B, H) – shouldn’t happen here, but guard anyway
            new_hxs = new_hxs.unsqueeze(0).expand(T, B, -1)
            new_cxs = new_cxs.unsqueeze(0).expand(T, B, -1)
        elif new_hxs.dim() == 3:
            # Either (B, T, H) or (T, B, H)
            if new_hxs.shape == (B, T, new_hxs.size(-1)):
                new_hxs = new_hxs.transpose(0, 1)  # (T, B, H)
                new_cxs = new_cxs.transpose(0, 1)  # (T, B, H)
            # else assume already (T, B, H)

        """
        logprobs, values, and entropy must not be detached because PPO uses
        them in the loss:

        - logprobs → policy gradient
        - values → value loss
        - entropy → entropy bonus
        - ar_loss - LSTM activation regularization
        - tar_loss - LSTM temporal activation regularization
        """
        return PolicyEvalOutput(
            values=values,  # (T, B)
            logprobs=logprobs,  # (T, B)
            entropy=entropy,  # (T, B)
            new_hxs=new_hxs,  # (T, B, H)
            new_cxs=new_cxs,  # (T, B, H)
            gates=gates,
            ar_loss=policy_output.ar_loss,
            tar_loss=policy_output.tar_loss,
            pred_obs=pred_obs,
            pred_raw=pred_raw,
        )

    def evaluate_actions(self, out, actions):
        """
        Evaluate log-prob and entropy for a single timestep.

        out: PolicyOutput from forward()
            - logits: (B, A)
            - values: (B,)
        actions: (B,) or (B,1)
        """

        # Ensure shape (B,)
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions = actions.squeeze(-1)

        dist = self._dist_from_logits(out.logits)  # (B, A)

        logprobs = dist.log_prob(actions)  # (B,)
        entropy = dist.entropy()  # (B,)

        return PolicyEvalOutput(
            values=out.values,  # (B,)
            logprobs=logprobs,  # (B,)
            entropy=entropy,  # (B,)
            new_hxs=out.new_hxs,  # (B, H)
            new_cxs=out.new_cxs,  # (B, H)
            gates=out.gates,  # (B, H) or (B, 1, H)
            ar_loss=out.ar_loss,
            tar_loss=out.tar_loss,
        )

    def _dist_from_logits(self, logits):
        return Categorical(logits=logits)


class ZeroFeatureEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, obs):
        # obs: (B, 0) or (B, T, 0)
        if obs.dim() == 2:
            B = obs.size(0)
            return torch.zeros(B, self.out_dim, device=obs.device)  # (B, H)
        else:
            B, T, _ = obs.shape
            return torch.zeros(B, T, self.out_dim, device=obs.device)  # (B, T, H)
