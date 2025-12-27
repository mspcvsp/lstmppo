import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from .types import PPOConfig, VecPolicyInput, VecPolicyOutput


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

        # --- LN-LSTM with DropConnect ---
        base_lstm = nn.LSTM(
            input_size=cfg.enc_hidden_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
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

    def init_hidden(self, batch_size, device):

        h = torch.zeros(batch_size,
                        self.lstm.module.hidden_size,
                        device=device)

        c = torch.zeros(batch_size,
                        self.lstm.module.hidden_size,
                        device=device)
        return h, c

    # ---------------------------------------------------------
    # Core forward pass (returns activations + AR/TAR)
    # ---------------------------------------------------------
    def _forward_core(self, x, hxs, cxs):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        enc = self.encoder(x)
        out, (h_n, c_n) = self.lstm(enc,
                                    (hxs.unsqueeze(0),
                                     cxs.unsqueeze(0)))
        out = self.ln(out)

        # --- Activation Regularization (AR) ---
        ar_loss = (out.pow(2).mean()) * self.ar_coef

        # --- Temporal Activation Regularization (TAR) ---
        if out.size(1) > 1:
            tar_loss =\
                ((out[:, 1:, :] - out[:, :-1, :]).pow(2).mean()) *\
                self.tar_coef
        else:
            tar_loss = torch.tensor(0.0, device=out.device)

        # Squeeze if single-step
        if out.size(1) == 1:
            out = out[:, 0, :]

        return out, h_n.squeeze(0), c_n.squeeze(0), ar_loss, tar_loss

    # ---------------------------------------------------------
    # Dataclass-based forward
    # ---------------------------------------------------------
    def forward(self, inp: VecPolicyInput) -> VecPolicyOutput:

        core_out, new_hxs, new_cxs, ar_loss, tar_loss =\
            self._forward_core(inp.obs,
                               inp.hxs,
                               inp.cxs)

        if core_out.dim() == 3:
            B, T, H = core_out.shape
            flat = core_out.reshape(B * T, H)
            logits = self.actor(flat).view(B, T, -1)
            values = self.critic(flat).view(B, T)
        else:
            logits = self.actor(core_out)
            values = self.critic(core_out).squeeze(-1)

        return VecPolicyOutput(
            logits=logits,
            values=values,
            new_hxs=new_hxs,
            new_cxs=new_cxs,
            ar_loss=ar_loss,
            tar_loss=tar_loss,
        )

    def act(self,
            policy_input: VecPolicyInput):

        policy_output = self.forward(policy_input)

        dist = Categorical(logits=policy_output.logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions, logprobs, policy_output

    def evaluate_actions(self,
                         inp: VecPolicyInput,
                         actions):

        if actions.dim() == 3:
            actions = actions.squeeze(-1)

        policy_output = self.forward(inp.obs, inp.hxs, inp.cxs)
        dist = Categorical(logits=policy_output.logits)

        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        return logprobs, entropy, policy_output.values


class SiLU_LN_LSTM(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size

        """Front-end MLP with SiLU
        Note: Gymnasium observation spaces return a shape tuple,
        not a scalar dimension.
        """
        self.fc1 = nn.Linear(cfg.obs_shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.actfcn = nn.SiLU()

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # LayerNorm on LSTM output
        self.ln = nn.LayerNorm(cfg.hidden_size)

        self._init_weights()

    def _init_weights(self):

        for name, param in self.named_parameters():
            if "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        for name, param in self.lstm.named_parameters():
            # Xavier for input weights
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            # Orthogonal for recurrent weights
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            # Bias: zero + forget gate bias trick
            elif "bias" in name:
                param.data.fill_(0.0)
                H = self.hidden_size
                param.data[H:2*H] = 1.0  # forget gate bias

    def forward(self, obs, hxs, cxs):
        """
        obs: (N, obs_shape) or (T,B,obs_shape)
        hxs, cxs: (N,H) or (B,H)
        """
        if obs.dim() == 2:
            # (N,obs)
            x = self.actfcn(self.fc1(obs))
            x = self.actfcn(self.fc2(x))
            x = x.unsqueeze(1)  # (N,1,128)

            hxs = hxs.unsqueeze(0)
            cxs = cxs.unsqueeze(0)

            out, (new_hxs, new_cxs) = self.lstm(x, (hxs, cxs))
            out = out.squeeze(1)          # (N,H)
            out = self.ln(out)
            new_hxs = new_hxs.squeeze(0)
            new_cxs = new_cxs.squeeze(0)

        else:
            # (T,B,obs)
            T, B, _ = obs.shape
            x = obs.reshape(T * B, -1)
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = x.reshape(T, B, -1)

            hxs = hxs.unsqueeze(0)
            cxs = cxs.unsqueeze(0)

            out, (new_hxs, new_cxs) = self.lstm(x, (hxs, cxs))
            out = self.ln(out)            # (T,B,H)
            new_hxs = new_hxs.squeeze(0)
            new_cxs = new_cxs.squeeze(0)

        return out, new_hxs, new_cxs
