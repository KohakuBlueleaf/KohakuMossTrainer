"""RMSNorm, RMSNorm + per-sample modulation, gated residual; one rounding at the end."""

import torch
from torch import Tensor

from kohakumoss_kernels.backends.eager._acc import acc, acc_dtype

__all__ = ["rms_norm", "rms_norm_mod", "gated_residual"]


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """`x / rms(x) * weight` over the last dim."""
    dt = acc_dtype(x)
    xf = x.to(dt)
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * acc(weight, dt)
    return y.to(x.dtype)


def rms_norm_mod(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    eps: float,
) -> Tensor:
    """`rms_norm(x) * (1 + scale[seg]) + shift[seg]`; shift/scale are [B, D] per sample."""
    dt = acc_dtype(x)
    xf = x.to(dt)
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * acc(weight, dt)
    if scale is not None:
        y = y * (1.0 + acc(scale, dt).index_select(0, seg))
    if shift is not None:
        y = y + acc(shift, dt).index_select(0, seg)
    return y.to(x.dtype)


def gated_residual(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None, r: float
) -> Tensor:
    """`x * r + o * gate[seg]` (gate [B, D] per sample, or None for an ungated add)."""
    dt = acc_dtype(x, o)
    of = o.to(dt)
    if gate is not None:
        of = of * acc(gate, dt).index_select(0, seg)
    xf = x.to(dt) if r == 1.0 else x.to(dt) * r
    return (xf + of).to(x.dtype)
