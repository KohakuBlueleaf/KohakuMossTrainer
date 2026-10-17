"""Weights of the fused linear ops in any storage dtype (fp32 masters included): cast to
the activation dtype at the matmul, gradients back in the storage dtype.
See docs/internals/fp16.md.
"""

import torch
from torch import Tensor


def activation(y: Tensor) -> Tensor:
    """`y` in the matmul compute dtype: fp32 activations under autocast are cast to the
    autocast dtype (as autocast casts a linear's inputs); 16-bit ones pass through."""
    dev = y.device.type
    if y.dtype == torch.float32 and torch.is_autocast_enabled(dev):
        return y.to(torch.get_autocast_dtype(dev))
    return y


def operand(w: Tensor | None, like: Tensor) -> Tensor | None:
    """`w` as a contiguous matmul operand in `like`'s (16-bit) dtype."""
    if w is None:
        return None
    return w.to(like.dtype).contiguous()


def weight_grad(g: Tensor, x: Tensor, dtype: torch.dtype) -> Tensor:
    """g^T x for g [M, N], x [M, K] (16-bit): fp32 accumulation and output, rounded once
    to the weight's storage `dtype`."""
    return torch.mm(g.t(), x, out_dtype=torch.float32).to(dtype)
