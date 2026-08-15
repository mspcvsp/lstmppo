import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.models.base_obs_encoder import BaseEncoder


class CNNObsEncoder(BaseEncoder):
    """
    Simple Dreamer‑style CNN encoder for 64x64x3 images.
    Output: (B, embed_dim)
    """

    def __init__(self, obs_space: gym.Space, embed_dim: int):
        super().__init__()
        assert isinstance(obs_space, gym.spaces.Box)
        assert len(obs_space.shape) == 3
        c = obs_space.shape[2]
        assert c == 3, "CNNObsEncoder expects RGB images"

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # 64x64 -> 4x4 after 4 stride‑2 convs
        self.fc = nn.Linear(256 * 4 * 4, embed_dim)

        self.apply(self._init_weights)

        self._output_size: int = embed_dim

    @property
    def output_size(self) -> int:
        return self._output_size

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, 64, 64, 3) float32 in [0,1]
        """
        x = obs.permute(0, 3, 1, 2)  # (B, 3, 64, 64)

        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
