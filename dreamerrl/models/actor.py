import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """
    Dreamer‑V3 policy network.
    Consumes factored z and deterministic h.
    Outputs logits over discrete actions.
    """

    def __init__(self, *, latent, net, probe=None):
        super().__init__()

        self.latent = latent
        self.net_cfg = net
        self.probe = probe

        self.z_embed = nn.Linear(latent.stoch_size, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.action_dim),
        )

    def forward(self, h, z):
        """
        h: (B, deter_size)
        z: (B, K, C)
        """
        z_e = self.z_embed(z)  # (B, K, H)
        z_sum = z_e.sum(dim=1)  # (B, H)
        h_e = self.h_embed(h)  # (B, H)

        features = h_e + z_sum
        logits = self.net(features)

        if self.probe:
            self.probe.actor(logits)

        return logits  # logits over actions

    @torch.no_grad()
    def act(self, state, deterministic: bool = False):
        """
        Sample an action from the policy.

        state: RSSMState with fields:
            - h: (B, deter_size)
            - z: (B, K, C)

        Returns:
            actions: (B,) int64
            logits:  (B, action_dim)
        """
        h = state.h
        z = state.z

        logits = self.forward(h, z)  # (B, action_dim)

        if deterministic:
            # Greedy action (used in evaluation or reproducibility tests)
            actions = torch.argmax(logits, dim=-1)
        else:
            # Categorical sampling
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1).to(logits.device)

        return actions, logits


def act_in_imagination(logits, deterministic_imagination: bool = False):
    if deterministic_imagination:
        a = logits.argmax(dim=-1)
    else:
        dist = Categorical(logits=logits)
        a = dist.sample().to(logits.device)

    return a
