import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from .types import PPOConfig, PolicyInput, PolicyOutput
from .types import PolicyEvalInput, PolicyEvalOutput


class WeightDrop(nn.Module):
    """
    Applies DropConnect to the recurrent weights of an RNN module.
    """
    def __init__(self,
                 module,
                 weights,
                 dropout):

        super().__init__()

        self.module = module
        self.weights = weights
        self.dropout = dropout

        # Save original parameters
        for w in weights:

            param = getattr(module, w)

            self.register_parameter(f"{w}_raw",
                                    nn.Parameter(param.data))

            # Remove original parameter from module
            del module._parameters[w]

    def _setweights(self):

        for w in self.weights:

            raw = getattr(self, f"{w}_raw")

            dropped = F.dropout(raw,
                                p=self.dropout,
                                training=self.training)

            setattr(self.module,
                    w,
                    dropped)

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module(*args, **kwargs)
    

class LSTMPPOPolicy(nn.Module):

    def __init__(self,
                 cfg: PPOConfig):

        super().__init__()

        self.ar_coef = cfg.lstm_ar_coef
        self.tar_coef = cfg.lstm_tar_coef
        self.device = cfg.device

        # --- SiLU encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_shape[0],
                      cfg.enc_hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.enc_hidden_size,
                      cfg.enc_hidden_size),
            nn.SiLU(),
        )

        for name, param in self.encoder.named_parameters():

            if "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        # --- LN-LSTM with DropConnect ---
        base_lstm = nn.LSTM(
            input_size=cfg.enc_hidden_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        for name, param in base_lstm.named_parameters():

            # Xavier for input weights
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            # Orthogonal for recurrent weights
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            # Bias: zero + forget gate bias trick
            elif "bias" in name:
                param.data.fill_(0.0)
                H = cfg.lstm_hidden_size
                param.data[H:2*H] = 1.0  # forget gate bias

        self.lstm = WeightDrop(
            base_lstm,
            weights=["weight_hh_l0"],
            dropout=cfg.dropconnect_p,
        )

        self.ln = nn.LayerNorm(cfg.lstm_hidden_size)

        # --- Heads ---
        self.actor = nn.Linear(cfg.lstm_hidden_size,
                               cfg.action_dim)
        
        self.critic = nn.Linear(cfg.lstm_hidden_size, 1)

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

        # Normalize input shape to (B, T, F)
        if x.dim() == 2:
            # (B, F) -> (B, 1, F)
            x = x.unsqueeze(1)

        enc = self.encoder(x)  # encoder must handle (B, T, F)

        # LSTM expects (B, T, F) with batch_first=True
        out, (h_n, c_n) = self.lstm(
            enc,
            (hxs.unsqueeze(0), cxs.unsqueeze(0)),  # (1, B, H)
        )

        out = self.ln(out)  # (B, T, H)

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

        return out, h_n.squeeze(0), c_n.squeeze(0), ar_loss, tar_loss

    # ---------------------------------------------------------
    # Dataclass-based forward
    # ---------------------------------------------------------
    def forward(self, inp: PolicyInput) -> PolicyOutput:
        """
        inp.obs: (B, obs_dim) OR (B, T, obs_dim)
        inp.hxs, inp.cxs: (B, H)
        """

        core_out, new_hxs, new_cxs, ar_loss, tar_loss = \
            self._forward_core(inp.obs, inp.hxs, inp.cxs)
        # core_out is ALWAYS (B, T, H)

        B, T, H = core_out.shape
        flat = core_out.reshape(B * T, H)

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
            new_hxs=new_hxs,
            new_cxs=new_cxs,
            ar_loss=ar_loss,
            tar_loss=tar_loss,
        )

    def act(self,
            policy_input: PolicyInput):

        policy_output = self.forward(policy_input)

        dist = Categorical(logits=policy_output.logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions, logprobs, policy_output

    def evaluate_actions_sequence(self,
                                  inp: PolicyEvalInput):
        """
        Fully sequence-aware PPO evaluation.
        Uses the same sequence-aware forward() used by act().

        obs: (T, B, obs_dim)
        hxs, cxs: (B, H)
        actions: (T, B) or (T, B, 1)
        """
        T, B = inp.obs.shape[0], inp.obs.shape[1]

        # Reorder to (B, T, obs_dim) for batch_first LSTM
        obs_bt = inp.obs.transpose(0, 1)

        # Build PolicyInput for sequence mode
        policy_input = PolicyInput(
            obs=obs_bt,   # (B, T, obs_dim)
            hxs=inp.hxs,  # (B, H)
            cxs=inp.cxs,  # (B, H)
        )

        # Forward pass (sequence-aware)
        policy_output = self.forward(policy_input)
        # logits: (B, T, A)
        # values: (B, T)

        # Convert back to (T, B, ...)
        logits = policy_output.logits.transpose(0, 1)   # (T, B, A)
        values = policy_output.values.transpose(0, 1)   # (T, B)

        # Actions: (T, B, 1) → (T, B)
        if actions.dim() == 3 and actions.size(-1) == 1:
            actions = actions.squeeze(-1)

        dist = Categorical(logits=logits)

        logprobs = dist.log_prob(actions)   # (T, B)
        entropy = dist.entropy().mean()     # scalar

        return PolicyEvalOutput(
            values=values,                  # (T, B)
            logprobs=logprobs,              # (T, B)
            entropy=entropy,                # scalar
            new_hxs=policy_output.new_hxs,  # (B, H)
            new_cxs=policy_output.new_cxs,  # (B, H)
            ar_loss=policy_output.ar_loss,  # scalar
            tar_loss=policy_output.tar_loss # scalar
        )
