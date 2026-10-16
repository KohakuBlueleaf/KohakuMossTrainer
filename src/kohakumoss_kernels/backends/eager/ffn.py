"""SiTU-GLU activation of the fused up/gate projection."""

import torch
from torch import Tensor

from kohakumoss_kernels.backends.eager._acc import acc_dtype

__all__ = ["situ_glu"]


def situ_glu(x: Tensor) -> Tensor:
    """x [N, 2F] = [up | gate] -> (4 tanh(g/4) sigmoid(g)) * (25 tanh(u/25)), [N, F]."""
    u, g = x.to(acc_dtype(x)).chunk(2, -1)
    a = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))
    return a.to(x.dtype)
