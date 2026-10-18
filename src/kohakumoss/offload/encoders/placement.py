"""The per-pass host offload shared by the frozen-encoder placements (text encoder and
VAE): host copies moved onto the device for one pass and released after it. See
docs/internals/offload.md."""

from collections.abc import Iterable

import torch
from torch import Tensor

from kohakumoss.offload.weights.layers import host_copy, tensor_bytes


class HostCopies:
    """Host copies of `tensors` (pinned when `pinned`; the tensors point at them between
    passes): `begin()` moves them onto `device` for a pass, `end()` points the tensors
    back at the host copies. `nbytes` is one pass's H2D bytes, `h2d_bytes` the total."""

    def __init__(self, tensors: Iterable[Tensor], device: torch.device, pinned: bool):
        self.device = device
        self.pairs: list[tuple[Tensor, Tensor]] = []
        for t in tensors:
            t.data = host_copy(t, pinned)
            self.pairs.append((t, t.data))
        self.nbytes = tensor_bytes(h for _, h in self.pairs)
        self.h2d_bytes = 0

    def begin(self) -> None:
        for t, h in self.pairs:
            t.data = h.to(self.device, non_blocking=True)
        self.h2d_bytes += self.nbytes

    def end(self) -> None:
        for t, h in self.pairs:
            t.data = h
