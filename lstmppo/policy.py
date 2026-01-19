import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from .types import Config, PolicyInput, PolicyOutput, LSTMCoreOutput
from .types import PolicyEvalInput, PolicyEvalOutput, LSTMGates
from .obs_encoder import build_obs_encoder


class GateLSTMCell(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 dropconnect_p,
                 bias=True):
        
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropconnect_p = dropconnect_p

        # Input weights
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size,
                                                   input_size))

        # Recurrent weights (raw)
        self.weight_hh_raw = nn.Parameter(torch.Tensor(4 * hidden_size,
                                                       hidden_size))

        # Biases
        if bias:
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

        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh_raw)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

        # Forget gate bias trick
        H = self.hidden_size
        self.bias_ih.data[H:2*H] = 1.0

    def _apply_dropconnect(self):

        if self.training:
    
            mask = torch.ones_like(self.weight_hh_raw)
    
            mask = F.dropout(mask,
                             p=self.dropconnect_p,
                             training=True)
            self.weight_hh = self.weight_hh_raw * mask

        else:
            self.weight_hh = self.weight_hh_raw

    def forward(self, x, hx):

        h, c = hx  # each (B, H)

        self._apply_dropconnect()

        gates = (
            x @ self.weight_ih.t()
            + h @ self.weight_hh.t()
            + self.bias_ih
            + self.bias_hh
        )

        H = self.hidden_size
        i = torch.sigmoid(gates[:, :H])
        f = torch.sigmoid(gates[:, H:2*H])
        g = torch.tanh(gates[:, 2*H:3*H])
        o = torch.sigmoid(gates[:, 3*H:4*H])

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)

        return new_h, new_c, (i, f, g, o)


class LSTMPPOPolicy(nn.Module):

    def __init__(self,
                 cfg: Config):

        super().__init__()

        self.ar_coef = cfg.lstm.lstm_ar_coef
        self.tar_coef = cfg.lstm.lstm_tar_coef

        self.obs_encoder = build_obs_encoder(cfg.env.obs_space,
                                             cfg.obs_dim)

        # --- SiLU encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim,
                      cfg.lstm.enc_hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.lstm.enc_hidden_size,
                      cfg.lstm.enc_hidden_size),
            nn.SiLU(),
        )

        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # --- LN-LSTM with DropConnect ---
        self.lstm_cell = GateLSTMCell(
            input_size=cfg.lstm.enc_hidden_size,    
            hidden_size=cfg.lstm.lstm_hidden_size,
            dropconnect_p=cfg.lstm.dropconnect_p
        )

        self.ln = nn.LayerNorm(cfg.lstm.lstm_hidden_size)

        # --- Heads ---
        self.actor = nn.Linear(cfg.lstm.lstm_hidden_size,
                               cfg.env.action_dim)
        
        self.critic = nn.Linear(cfg.lstm.lstm_hidden_size, 1)

        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.xavier_uniform_(self.critic.weight)
    
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    # ---------------------------------------------------------
    # Core forward pass (returns activations + AR/TAR)
    # ---------------------------------------------------------
    def _forward_core(self, x, hxs, cxs):
        """
        _forward_core performs the actual LSTM unroll and therefore produces
        batch‑first tensors shaped (B, T, H). This is the natural format for:

        - the encoder (batch-first)
        - the LSTM cell (batch-first)
        - the policy/value heads (batch-first)

        During the unroll we also record the full per‑timestep sequences of:

        - gate activations  (i, f, g, o)
        - hidden states     (h_t)
        - cell states       (c_t)

        These sequences are required for LSTM diagnostics such as:

        - gate drift
        - gate saturation
        - gate entropy
        - hidden/cell state drift
        - hidden/cell saturation

        All tensors returned here remain batch‑first (B, T, H). They will be
        transposed later in evaluate_actions_sequence to match PPO’s
        time‑major (T, B, ...) rollout format.
        """
        obs_flat = self.obs_encoder(x)  # x may be dict/tuple/box

        if obs_flat.dim() == 2:
            obs_flat = obs_flat.unsqueeze(1)

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
            h, c, gates = self.lstm_cell(enc[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
            gate_list.append(gates)

            # store per‑timestep hidden and cell states [B, H]
            h_list.append(h) 
            c_list.append(c)

        out = torch.cat(outputs, dim=1)  # (B, T, H)
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
            tar_loss = (
                (out[:, 1:, :] - out[:, :-1, :]).pow(2).mean()
                * self.tar_coef
            )
        else:
            tar_loss = torch.tensor(0.0, device=out.device)
        
        return LSTMCoreOutput(out=out,
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
                              ))

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
        logits = self.actor(flat).view(B, T, -1)   # (B, T, A)
        values = self.critic(flat).view(B, T)      # (B, T)

        # If single-step, squeeze back to (B, A) and (B,)
        if T == 1:
            logits = logits[:, 0, :]
            values = values[:, 0]

        # logits: (B, T, A) for sequences, (B, A) for single-step
        # values: (B, T) or (B,)
        return PolicyOutput(
            logits=logits,
            values=values,
            new_hxs=core_out.h,
            new_cxs=core_out.c,
            ar_loss=core_out.ar_loss,
            tar_loss=core_out.tar_loss,
            gates=core_out.gates
        )

    def act(self,
            policy_input: PolicyInput):

        policy_output = self.forward(policy_input)

        dist = Categorical(logits=policy_output.logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions, logprobs, policy_output

    def evaluate_actions_sequence(self,
                                  inp: PolicyEvalInput) -> PolicyEvalOutput:
        """
        Fully sequence-aware PPO evaluation.

        inp.obs:     (T, B, obs_dim)
        inp.hxs:     (B, H)
        inp.cxs:     (B, H)
        inp.actions: (T, B) or (T, B, 1)
        """
        T, B = inp.obs.shape[0], inp.obs.shape[1]

        # (T, B, F) -> (B, T, F) for batch_first LSTM
        obs_bt = inp.obs.transpose(0, 1)  # (B, T, obs_dim)

        policy_input = PolicyInput(
            obs=obs_bt,    # (B, T, obs_dim)
            hxs=inp.hxs,   # (B, H)
            cxs=inp.cxs,   # (B, H)
        )

        # Forward through encoder + LSTM + heads
        policy_output = self.forward(policy_input)
        # policy_output.logits: (B, T, A)
        # policy_output.values: (B, T)

        # Back to (T, B, ...)
        logits = policy_output.logits.transpose(0, 1)  # (T, B, A)
        values = policy_output.values.transpose(0, 1)  # (T, B)

        # Actions: ensure shape (T, B)
        if inp.actions.dim() == 3 and inp.actions.size(-1) == 1:
            actions = inp.actions.squeeze(-1)          # (T, B)
        else:
            actions = inp.actions                      # (T, B)

        dist = Categorical(logits=logits)

        assert logits.dim() == 3,\
            f"logits must be (T,B,A), got {logits.shape}"
        
        assert actions.shape == (T, B),\
            f"actions must be (T,B), got {actions.shape}"

        # Correct shape: (T, B)
        logprobs = dist.log_prob(actions)     # (T, B)
        entropy = dist.entropy()              # (T, B)

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
        gates_t = policy_output.gates.transposed()

        i_gates = gates_t.i_gates
        f_gates = gates_t.f_gates
        g_gates = gates_t.g_gates
        o_gates = gates_t.o_gates
        c_gates = gates_t.c_gates
        h_gates = gates_t.h_gates

        """
        hxs and cxs are recurrent state outputs from the LSTM that are fed
        into the next rollout step. Detach to prevent gradients flowing
        across rollout boundaries. PPO treats each rollout as a truncated
        BPTT segment
        """
        new_hxs = policy_output.new_hxs.transpose(0, 1).detach()  # (T, B, H)
        new_cxs = policy_output.new_cxs.transpose(0, 1).detach()  # (T, B, H)

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
            values=values,          # (T, B)
            logprobs=logprobs,      # (T, B)
            entropy=entropy,        # (T, B)
            new_hxs=new_hxs,        # (T, B, H)
            new_cxs=new_cxs,        # (T, B, H)
            gates=LSTMGates(
                i_gates=i_gates,    # (T, B, H)
                f_gates=f_gates,
                g_gates=g_gates,
                o_gates=o_gates,
                c_gates=c_gates,
                h_gates=h_gates,
            ).detached,
            ar_loss=policy_output.ar_loss,
            tar_loss=policy_output.tar_loss,
        )
