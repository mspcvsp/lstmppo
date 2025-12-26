import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class VecPolicyInput:
    obs: torch.Tensor          # (B, obs_dim) or (T,B,obs_dim)
    hxs: torch.Tensor          # (B, hidden)
    cxs: torch.Tensor          # (B, hidden)

class WeightDrop(nn.Module):
    """
    Applies DropConnect to the recurrent weights of an RNN module.
    """
    def __init__(self, module, weights, dropout):
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
            dropped = F.dropout(raw, p=self.dropout, training=self.training)
            setattr(self.module, w, dropped)

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module(*args, **kwargs)


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
        obs: (N, obs_dim) or (T,B,obs_dim)
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


class LSTMPPOPolicy(nn.Module):
    
    def __init__(self,
                 cfg):

        super().__init__()
        self.hidden_size = cfg.hidden_size

        self.core = SiLU_LN_LSTM(cfg)

        # Actor and critic heads
        self.actor = nn.Linear(self.hidden_size,
                               cfg.action_dim)

        self.critic = nn.Linear(self.hidden_size, 1)

        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

        nn.init.xavier_uniform_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self,
                obs,
                hxs,
                cxs):
        """
        obs: (N,obs) or (T,B,obs)
        hxs,cxs: (N,H) or (B,H)
        """
        core_out, new_hxs, new_cxs = self.core(obs, hxs, cxs)

        logits = self.actor(core_out)
        values = self.critic(core_out).squeeze(-1)

        return logits, values, new_hxs, new_cxs

    def act(self,
            obs,
            hxs,
            cxs):

        logits, values, new_hxs, new_cxs = self.forward(obs, hxs, cxs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions, logprobs, values, new_hxs, new_cxs

    def evaluate_actions(self,
                         obs,
                         hxs,
                         cxs,
                         actions):

        if actions.dim() == 3:
            actions = actions.squeeze(-1)

        logits, values, _, _ = self.forward(obs, hxs, cxs)
        dist = Categorical(logits=logits)

        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        return logprobs, entropy, values