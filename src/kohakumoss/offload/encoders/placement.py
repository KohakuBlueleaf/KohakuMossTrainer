"""The per-pass host offload shared by the frozen-encoder placements (text encoder and
VAE): host copies moved onto the device for one pass and released after it. See
docs/internals/offload.md."""

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor

from kohakumoss.offload.weights.layers import device_dtype, host_copy, tensor_bytes


class HostCopies:
    """The host form of `tensors` (the tensors point at it between passes): `host`
    "copy" makes host copies in each tensor's device dtype (pinned when `pinned`),
    "file" keeps the tensors as they are (the checkpoint's file pages). `dtypes` maps a
    tensor id to its device dtype where its host form holds another (`device_dtype`).
    `begin()` moves them onto `device` for a pass, `end()` points the tensors back at
    the host form, `release()` drops it. `nbytes` is one pass's H2D bytes, `h2d_bytes`
    the total."""

    def __init__(
        self,
        tensors: Iterable[Tensor],
        device: torch.device,
        pinned: bool,
        host: str = "copy",
        dtypes: Mapping[int, torch.dtype] | None = None,
    ):
        self.device = device
        self.pairs: list[tuple[Tensor, Tensor, torch.dtype]] = []
        for t in tensors:
            want = device_dtype(t, dtypes)
            if host == "copy":
                t.data = host_copy(t, pinned, want)
            self.pairs.append((t, t.data, want))
        self.nbytes = tensor_bytes(h for _, h, _ in self.pairs)
        self.h2d_bytes = 0

    def begin(self) -> None:
        for t, h, want in self.pairs:
            t.data = h.to(self.device, want, non_blocking=True)
        self.h2d_bytes += self.nbytes

    def end(self) -> None:
        for t, h, _ in self.pairs:
            t.data = h

    def release(self) -> None:
        """Drop the host form (the caller re-points the tensors)."""
        self.pairs = []
