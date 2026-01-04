import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


class FlatObsEncoder(nn.Module):

    def __init__(self,
                 space: gym.spaces.Box):
        
        super().__init__()
        
        self.output_size = int(np.prod(space.shape))

    def forward(self, obs):
        return obs.view(obs.shape[0], -1)


class DictObsEncoder(nn.Module):
    
    def __init__(self,
                 space: gym.spaces.Dict):
        
        super().__init__()
        
        self.keys = list(space.spaces.keys())

        self.encoders = nn.ModuleDict()

        for k, subspace in space.spaces.items():
            
            if isinstance(subspace,
                          gym.spaces.Box):

                self.encoders[k] = FlatObsEncoder(subspace)
            else:
                raise NotImplementedError(
                    f"Unsupported subspace: {subspace}"
                )

        self.output_size =\
            sum(enc.output_size for enc in self.encoders.values())

    def forward(self, obs_dict):
        encoded = [self.encoders[k](obs_dict[k]) for k in self.keys]
        return torch.cat(encoded, dim=-1)


class TupleObsEncoder(nn.Module):

    def __init__(self,
                 space: gym.spaces.Tuple):
        
        super().__init__()
        
        self.encoders = nn.ModuleList(
            [FlatObsEncoder(s) for s in space.spaces]
        )
        
        self.output_size = sum(enc.output_size for enc in self.encoders)

    def forward(self, obs_tuple):
        encoded = [enc(o) for enc, o in zip(self.encoders, obs_tuple)]
        return torch.cat(encoded, dim=-1)


def build_obs_encoder(space):

    if isinstance(space, gym.spaces.Box):
        return FlatObsEncoder(space)
    elif isinstance(space, gym.spaces.Dict):
        return DictObsEncoder(space)
    elif isinstance(space, gym.spaces.Tuple):
        return TupleObsEncoder(space)
    else:
        raise NotImplementedError(f"Unsupported observation space: {space}")