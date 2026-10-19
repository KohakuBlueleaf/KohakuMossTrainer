"""A host form held as a quantized base: the copy that moves the tensor to a device
moves the base's buffers and builds the dense tensor there (`base.weight(dtype)`),
so the host keeps the stored codes, not the dense weight. See
docs/internals/offload.md#text-encoder."""

import copy

import torch
from torch import Tensor, nn


class Deferred:
    """The host form of one dense module tensor of `shape`, built from `base` (a
    quantized base with `weight(dtype)`); `nbytes` is what one copy moves."""

    def __init__(self, base: nn.Module, shape: torch.Size):
        self.base = base
        self.shape = torch.Size(shape)
        self.nbytes = sum(b.nbytes for b in base.buffers())

    def to(self, device, dtype: torch.dtype, non_blocking: bool = False) -> Tensor:
        """The dense tensor in `dtype`, built on `device` from the moved buffers."""
        moved = copy.copy(self.base)
        moved._buffers = {
            n: None if b is None else b.to(device, non_blocking=non_blocking)
            for n, b in self.base._buffers.items()
        }
        return moved.weight(dtype)


def placeholder(shape, dtype: torch.dtype) -> Tensor:
    """A host tensor of `shape` and `dtype` over one element (what a tensor whose host
    form is Deferred points at between passes)."""
    return torch.zeros((), dtype=dtype).expand(shape)


def put(dst: Tensor, src, device, non_blocking: bool) -> None:
    """Copy host form `src` (a tensor or a Deferred) into device tensor `dst`."""
    if isinstance(src, Deferred):
        src = src.to(device, dst.dtype, non_blocking)
    dst.copy_(src, non_blocking=non_blocking)
