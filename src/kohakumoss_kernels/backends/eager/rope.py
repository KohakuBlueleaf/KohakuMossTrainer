"""Per-head qk RMSNorm followed by multi-axis split-half RoPE from token positions."""

import torch
from torch import Tensor

from kohakumoss_kernels.backends.eager._acc import acc_dtype

__all__ = ["qk_norm_rope"]


def rope_sin_cos(
    pos: Tensor, bands: Tensor, dtype=torch.float32
) -> tuple[Tensor, Tensor]:
    """sin, cos [N, Dh/2]: dim j uses axis j // n_bands and band j % n_bands."""
    phase = (pos.to(dtype).unsqueeze(-1) * bands.to(dtype)).flatten(-2)
    return phase.sin(), phase.cos()


def _norm_rotate(x: Tensor, weight: Tensor, sin: Tensor, cos: Tensor, eps: float):
    dt = sin.dtype
    xf = x.to(dt)
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.to(dt)
    a, b = xf.chunk(2, -1)
    s, c = sin.unsqueeze(1), cos.unsqueeze(1)
    return torch.cat((a * c - b * s, b * c + a * s), -1).to(x.dtype)


def qk_norm_rope(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    pos: Tensor,
    bands: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """q, k [N, H, Dh]: RMSNorm over Dh with weight, then rotate pairs (j, j + Dh/2)."""
    sin, cos = rope_sin_cos(pos, bands, acc_dtype(q, k))
    return _norm_rotate(q, q_weight, sin, cos, eps), _norm_rotate(
        k, k_weight, sin, cos, eps
    )
