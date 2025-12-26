from dataclasses import dataclass
from typing import Any, List
import torch


@dataclass
class RolloutStep:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    hxs: torch.Tensor
    cxs: torch.Tensor


@dataclass
class RecurrentBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    hxs: torch.Tensor
    cxs: torch.Tensor


@dataclass
class VecPolicyInput:
    obs: torch.Tensor
    hxs: torch.Tensor
    cxs: torch.Tensor


@dataclass
class VecEnvState:
    obs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: List[Any]
    hxs: torch.Tensor
    cxs: torch.Tensor

    def to_policy_input(self,
                        detach: bool = False) -> VecPolicyInput:

        if detach:
            return VecPolicyInput(
                obs=self.obs,
                hxs=self.hxs.detach(),
                cxs=self.cxs.detach(),
            )
        
        return VecPolicyInput(self.obs,
                              self.hxs,
                              self.cxs)