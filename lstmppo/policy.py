import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from .types import Config, PolicyInput, PolicyOutput, LSTMCoreOutput
from .types import PolicyEvalInput, PolicyEvalOutput, LSTMGates
from .obs_encoder import build_obs_encoder


class WeightDrop(nn.Module):
    """
    Applies DropConnect to the recurrent weights of an RNN module.
    Respects .training so DropConnect is OFF in eval() mode.
    """
    def __init__(self,
                 module,
                 weights,
                 dropout):

        super().__init__()

        self.module = module
        self.weights = weights
        self.dropout = dropout

        # Save raw parameters
        for w in weights:
            
            param = getattr(module, w)

            self.register_parameter(f"{w}_raw",
                                    nn.Parameter(param.data))
            
            del module._parameters[w]

    def _setweights(self):
        """
        Inject dropped or raw weights depending on self.training.
        """
        for w in self.weights:

            raw = getattr(self, f"{w}_raw")

            if self.training:
                # DropConnect active
                dropped = F.dropout(raw,
                                    p=self.dropout,
                                    training=True)
            else:
                # Deterministic: use raw weights
                dropped = raw

            setattr(self.module, w, dropped)

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module(*args, **kwargs)


class GateLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size,
                                                   input_size))

        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size,
                                                   hidden_size))

        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

        # Forget gate bias trick
        H = self.hidden_size
        self.bias_ih.data[H:2*H] = 1.0

    def forward(self, x, hx):
        h, c = hx  # each (B, H)

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
        x: (B, obs_dim)       for single-step
        or (B, T, obs_dim)    for sequence
        hxs, cxs: (B, H) initial hidden state per sequence
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

        for t in range(T):
            h, c, gates = self.lstm_cell(enc[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
            gate_list.append(gates)

        out = torch.cat(outputs, dim=1)  # (B, T, H)
        i_gates, f_gates, g_gates, o_gates = zip(*gate_list)

        # Stack into (B, T, H)
        i_gates = torch.stack(i_gates, dim=1)
        f_gates = torch.stack(f_gates, dim=1)
        g_gates = torch.stack(g_gates, dim=1)
        o_gates = torch.stack(o_gates, dim=1)

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
                                  i_gates=i_gates.detach(),
                                  f_gates=f_gates.detach(),
                                  g_gates=g_gates.detach(),
                                  o_gates=o_gates.detach()
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
        Transpose to align gate driftwith:
        - values (T, B)
        - logprobs (T, B)
        - entropy (T, B)"
        """
        i_gates = policy_output.gates.i_gates.transpose(0, 1) # (T, B, H)
        f_gates = policy_output.gates.f_gates.transpose(0, 1)
        g_gates = policy_output.gates.g_gates.transpose(0, 1)
        o_gates = policy_output.gates.o_gates.transpose(0, 1)

        return PolicyEvalOutput(
            values=values,                    # (T, B)
            logprobs=logprobs,                # (T, B)
            entropy=entropy,                  # (T, B)
            new_hxs=policy_output.new_hxs,    # (B, H)
            new_cxs=policy_output.new_cxs,    # (B, H)
            ar_loss=policy_output.ar_loss,    # scalar
            tar_loss=policy_output.tar_loss,  # scalar
            gates=LSTMGates(i_gates=i_gates,
                            f_gates=f_gates,
                            g_gates=g_gates,
                            o_gates=o_gates)
        )