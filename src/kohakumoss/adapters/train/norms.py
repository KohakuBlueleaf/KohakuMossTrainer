"""Trained norm deltas: `weight = original + m * w_norm` through a parametrization
(the norm keeps reading `.weight`, so fused norm kernels see the adapted weight)."""

import torch
import torch.nn.utils.parametrize as parametrize
from torch import Tensor, nn


class NormDelta(nn.Module):
    """Parametrization adding a trainable fp32 delta to a 1-D norm parameter."""

    def __init__(self, dim: int, device=None):
        super().__init__()
        self.w_norm = nn.Parameter(torch.zeros(dim, device=device, dtype=torch.float32))
        self.multiplier = 1.0

    def forward(self, original: Tensor) -> Tensor:
        return (original.float() + self.w_norm * self.multiplier).to(original.dtype)


def attach(module: nn.Module, name: str = "weight") -> NormDelta:
    """Parametrize `module.<name>` with a zero-initialised NormDelta."""
    param = getattr(module, name)
    delta = NormDelta(param.shape[0], device=param.device)
    parametrize.register_parametrization(module, name, delta)
    return delta


def detach(module: nn.Module, bake: bool, name: str = "weight") -> None:
    """Remove the parametrization, baking the adapted value in when `bake`."""
    if parametrize.is_parametrized(module, name):
        parametrize.remove_parametrizations(module, name, leave_parametrized=bake)
