import torch
import torch.nn as nn

from dreamerrl.models.base_obs_encoder import BaseEncoder
from dreamerrl.utils.transforms import symlog


# ============================================================
# Dreamer ObsEncoder (MLP with SiLU + Xavier)
# ============================================================
class MLPObsEncoder(BaseEncoder):
    """
    Dreamer-style observation encoder.

    - Input:  (B, flat_dim) tensor
    - Output: (B, embed_dim) tensor
    - Uses SiLU activations and Xavier initialization.
    """

    def __init__(self, flat_dim: int, embed_dim: int = 256) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

        self._output_size: int = embed_dim

    @property
    def output_size(self) -> int:
        return self._output_size

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2:
            raise ValueError(...)
        obs = symlog(obs)
        return self.net(obs)
