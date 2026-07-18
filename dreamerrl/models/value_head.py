import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """
    Dreamer‑V3 value model.
    Consumes factored z and deterministic h.
    Outputs categorical logits over value bins.
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
            nn.Linear(net.hidden_size, net.value_bins),  # distribution over value bins
        )

        self.value_bins = net.value_bins
        # symmetric bin centers in symlog space
        self.bin_values = torch.linspace(-1.0, 1.0, self.value_bins)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
            self.probe.critic(logits)

        return logits

    def readout(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Turn categorical logits into scalar value by expectation over bin centers.
        """
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bin_values.to(logits.device)).sum(dim=-1)  # (B,)

    def loss(self, logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Distributional cross‑entropy loss for symlog‑transformed returns.
        """
        target = torch.sign(returns) * torch.log1p(torch.abs(returns))
        with torch.no_grad():
            bin_idx = torch.argmin(
                torch.abs(target.unsqueeze(-1) - self.bin_values.to(target.device)),
                dim=-1,
            )
        return F.cross_entropy(
            logits.view(-1, self.value_bins),
            bin_idx.view(-1),
        )

    def loss_from_logits(self, logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, returns)
