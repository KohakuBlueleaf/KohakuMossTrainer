"""DoRA epilogues on an already merged weight W + dW (multiplier 1)."""

from collections.abc import Callable

import torch
from lycoris.functional.general import weight_decompose
from torch import Tensor


def _eps_norm(norm: Tensor, dtype: torch.dtype) -> Tensor:
    return (norm + torch.finfo(dtype).eps).detach()


def detached_out(merged: Tensor, dora_scale: Tensor) -> Tensor:
    """Output-axis DoRA with the row norm treated as a constant in backward."""
    norm = merged.reshape(merged.shape[0], -1).norm(dim=1, keepdim=True)
    return merged * (dora_scale.to(merged.dtype) / _eps_norm(norm, merged.dtype))


def detached_in(merged: Tensor, dora_scale: Tensor) -> Tensor:
    """Input-axis DoRA with the column norm treated as a constant in backward."""
    norm = merged.norm(dim=0, keepdim=True)
    return merged * (dora_scale.to(merged.dtype) / _eps_norm(norm, merged.dtype))


def detached(fn: Callable, lyco, merged: Tensor) -> Tensor:
    """A detached DoRA epilogue with the layer's magnitude."""
    return fn(merged, lyco.dora_scale.float())


def decomposed(t, backend: str | None, merged: Tensor) -> Tensor:
    """LyCORIS weight_decompose on an already-merged weight (multiplier 1)."""
    scale = t.lyco.dora_scale.float()
    return weight_decompose(merged, scale, 1.0, t.wd_on_out, backend=backend)
