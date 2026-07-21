import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Unified interface for all Dreamer encoders."""

    output_size: int

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method must be implemented by subclasses.")
