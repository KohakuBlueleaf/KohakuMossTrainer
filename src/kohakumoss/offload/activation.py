"""Activation offload: saved tensors of a block live in host memory between fwd and bwd.

Each saved tensor above `min_bytes` is copied to (pinned) host memory on a D2H stream
during forward; in backward the first unpack of block i fetches all of block i's
packets and prefetches block i-1 on an H2D stream. Parameters and registered storages
(streamed weight slots) are never offloaded. The device / host-copy / prefetch variant
is bound once at construction.
"""

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Packet:
    """One offloaded saved tensor."""

    host: Tensor
    block: int
    d2h: object = None
    gpu: Tensor | None = None
    h2d: object = None


def _noop(block: int) -> None:
    return None


class ActivationOffload:
    """saved_tensors_hooks engine; `block(i)` scopes the hooks to block i."""

    def __init__(
        self,
        device: torch.device | str,
        min_bytes: int = 1 << 20,
        pinned: bool = True,
        prefetch: bool = True,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.min_bytes = min_bytes
        self.cuda = self.device.type == "cuda"
        self.pinned = pinned and self.cuda
        self.skip: set[int] = set()
        self.pending: dict[int, list[Packet]] = {}
        self.current = 0
        self.stats = {"packed_bytes": 0, "kept": 0, "packed": 0}
        if self.cuda:
            self.d2h_stream = torch.cuda.Stream(self.device)
            self.h2d_stream = torch.cuda.Stream(self.device)
            self._store = self._store_device
            self._load = self._load_device
            self._prefetch = self._fetch if prefetch else _noop
        else:
            self._store = self._store_host
            self._load = self._load_host

    def exclude(self, tensors: Iterable[Tensor]) -> None:
        """Never offload these tensors' storages (parameters, weight slots)."""
        for t in tensors:
            if t.numel():
                self.skip.add(t.untyped_storage().data_ptr())

    @contextmanager
    def block(self, index: int):
        prev = self.current
        self.current = index
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            yield
        self.current = prev

    def pack(self, t: Tensor):
        if (
            t.device != self.device
            or t.numel() * t.element_size() < self.min_bytes
            or t.untyped_storage().data_ptr() in self.skip
        ):
            self.stats["kept"] += 1
            return t
        self.stats["packed"] += 1
        self.stats["packed_bytes"] += t.numel() * t.element_size()
        pk = self._store(t)
        self.pending.setdefault(self.current, []).append(pk)
        return pk

    def unpack(self, pk):
        if isinstance(pk, Tensor):
            return pk
        return self._load(pk)

    def _store_host(self, t: Tensor) -> Packet:
        return Packet(t.detach().clone(), self.current)

    def _load_host(self, pk: Packet) -> Tensor:
        self.pending.pop(pk.block, None)
        return pk.host

    def _store_device(self, t: Tensor) -> Packet:
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=self.pinned)
        self.d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.d2h_stream):
            host.copy_(t, non_blocking=True)
        t.record_stream(self.d2h_stream)
        ev = torch.cuda.Event()
        ev.record(self.d2h_stream)
        return Packet(host, self.current, d2h=ev)

    def _fetch(self, block: int) -> None:
        for pk in self.pending.pop(block, ()):
            if pk.gpu is not None:
                continue
            self.h2d_stream.wait_event(pk.d2h)
            with torch.cuda.stream(self.h2d_stream):
                gpu = torch.empty(
                    pk.host.shape, dtype=pk.host.dtype, device=self.device
                )
                gpu.copy_(pk.host, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.h2d_stream)
            pk.gpu, pk.h2d = gpu, ev

    def _load_device(self, pk: Packet) -> Tensor:
        if pk.gpu is None:
            self.pending.setdefault(pk.block, []).append(pk)
            self._fetch(pk.block)
            self._prefetch(pk.block - 1)
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(pk.h2d)
        gpu, pk.gpu = pk.gpu, None
        gpu.record_stream(cur)
        return gpu

    def reset(self) -> None:
        """Drop packets of a step that never ran backward."""
        self.pending.clear()
