"""RMSNorm parameter holder; the math runs through the OpSet with this module's eps."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Learned per-channel weight and an `eps` the residual fold rescales."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def init_weights(self) -> None:
        nn.init.ones_(self.weight)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps:g}"
