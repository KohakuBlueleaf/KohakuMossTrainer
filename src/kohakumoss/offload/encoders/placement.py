"""The per-pass host offload shared by the frozen-encoder placements (text encoder and
VAE): host copies moved onto the device for one pass and released after it. See
docs/internals/offload.md."""

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor

from kohakumoss.offload.host.arena import host_copies
from kohakumoss.offload.weights.deferred import Deferred
from kohakumoss.offload.weights.layers import device_dtype, tensor_bytes


class HostCopies:
    """The host form of `tensors` (the tensors point at it between passes): `host`
    "copy" makes host copies in each tensor's device dtype in one HostArena (pinned
    when `pinned`),
    "file" keeps the tensors as they are (the checkpoint's file pages); `dtypes` maps a
    tensor id to its device dtype where its host form holds another, `deferred` to a
    Deferred host form. `begin()` moves them onto `device` for a pass, `end()` points
    the tensors back at the host form, `release()` drops it. `nbytes` is one pass's H2D
    bytes, `h2d_bytes` the total."""

    def __init__(
        self,
        tensors: Iterable[Tensor],
        device: torch.device,
        pinned: bool,
        host: str = "copy",
        dtypes: Mapping[int, torch.dtype] | None = None,
        deferred: Mapping[int, Deferred] | None = None,
    ):
        deferred = deferred or {}
        tensors = list(tensors)
        self.device = device
        copied = [t for t in tensors if host == "copy" and id(t) not in deferred]
        wants = [device_dtype(t, dtypes) for t in copied]
        self.arena, _ = host_copies(copied, pinned, device, wants)
        self.pairs: list[tuple[Tensor, Tensor, torch.dtype]] = []
        self.sources: list = []
        for t in tensors:
            self.pairs.append((t, t.data, device_dtype(t, dtypes)))
            self.sources.append(deferred.get(id(t), t.data))
        self.nbytes = tensor_bytes(self.sources)
        self.h2d_bytes = 0

    def begin(self) -> None:
        for (t, _, want), src in zip(self.pairs, self.sources):
            t.data = src.to(self.device, want, non_blocking=True)
        self.h2d_bytes += self.nbytes

    def end(self) -> None:
        for t, h, _ in self.pairs:
            t.data = h

    def release(self) -> None:
        """Drop the host form (the caller re-points the tensors)."""
        self.pairs = []
        self.sources = []
        self.arena.release()
