"""Accumulation dtype of the eager reference: at least fp32, fp64 for fp64 references."""

import torch
from torch import Tensor


def acc_dtype(*tensors: Tensor | None) -> torch.dtype:
    dtype = torch.float32
    for t in tensors:
        if t is not None:
            dtype = torch.promote_types(dtype, t.dtype)
    return dtype


def acc(t: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    return None if t is None else t.to(dtype)
