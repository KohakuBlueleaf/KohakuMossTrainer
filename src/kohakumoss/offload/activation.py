"""Activation offload: saved tensors of a block live in host memory between forward
and backward, prefetched per block; a saved dtype cast of a parameter is recast in
backward instead, and one tensor saved twice in a block is one packet. See
docs/internals/offload.md."""

import weakref
from collections import deque
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor

from kohakumoss.offload.accel import module, streamed, with_index
from kohakumoss.offload.host.slab import HostSlab
from kohakumoss_common.calls import nothing

CAST = "ToCopyBackward0"
VIEWS = frozenset(
    {
        "TBackward0",
        "TransposeBackward0",
        "PermuteBackward0",
        "ViewBackward0",
        "UnsafeViewBackward0",
        "AliasBackward0",
    }
)


@dataclass
class Packet:
    """One offloaded saved tensor; `users` saved variables share it. `given`: its host
    buffer went back to the pool (`device_copy` is then the only copy)."""

    host: Tensor
    block: int
    d2h: object = None
    device_copy: Tensor | None = None
    h2d: object = None
    flat: Tensor | None = None
    users: int = 1
    given: bool = False


@dataclass
class Recast:
    """A saved view of a dtype cast of parameter `param` (its value at `version`):
    cast again and viewed with the saved shape / stride / offset on unpack."""

    param: Tensor
    dtype: torch.dtype
    version: int
    shape: torch.Size
    stride: tuple[int, ...]
    offset: int


def cast_source(t: Tensor) -> Tensor | None:
    """The contiguous leaf parameter whose dtype cast `t` is, or a view of (autocast's
    weight cast, `w.t()` as a linear saves it), else None."""
    fn = t.grad_fn
    while fn is not None and fn.name() in VIEWS:
        fn = fn.next_functions[0][0]
    if fn is None or fn.name() != CAST:
        return None
    param = getattr(fn.next_functions[0][0], "variable", None)
    if param is None or param.device != t.device or not param.is_contiguous():
        return None
    if t.untyped_storage().nbytes() != param.numel() * t.element_size():
        return None
    return param


def recast_of(t: Tensor, param: Tensor) -> Recast:
    return Recast(
        param, t.dtype, param._version, t.shape, t.stride(), t.storage_offset()
    )


def _recast(pk: Recast) -> Tensor:
    """The parameter cast again; raises when the parameter changed since forward."""
    if pk.param._version != pk.version:
        raise RuntimeError(
            "activation offload: a parameter whose dtype cast was saved for backward "
            "was modified in place before that backward"
        )
    cast = pk.param.detach().to(pk.dtype)
    return cast.as_strided(pk.shape, pk.stride, pk.offset)


def _key(t: Tensor) -> tuple:
    return (
        t.untyped_storage().data_ptr(),
        t.storage_offset(),
        tuple(t.shape),
        t.stride(),
        t.dtype,
        t._version,
    )


class ActivationOffload:
    """saved_tensors_hooks engine; `block(i)` scopes the hooks to block i. `inflight`
    blocks of copies (each direction) may be queued on the side streams; a copy of a
    newer block first waits on the host for the oldest block's copies."""

    def __init__(
        self,
        device: torch.device | str,
        min_bytes: int = 1 << 20,
        pinned: bool = True,
        prefetch: bool = True,
        inflight: int = 2,
    ):
        self.device = with_index(device)
        self.min_bytes = min_bytes
        self.streamed = streamed(self.device)
        self.dm = module(self.device)
        self.pinned = pinned and self.streamed
        self.inflight = inflight
        self.skip: set[int] = set()
        self.seen: dict[tuple, tuple] = {}
        self.pending: dict[int, list[Packet]] = {}
        sync = partial(self.dm.synchronize, self.device) if self.streamed else nothing
        self.slab = HostSlab(self.device, self.pinned, sync)
        self.queued: dict[str, deque] = {"d2h": deque(), "h2d": deque()}
        self.current = 0
        self.stats = {
            "packed_bytes": 0,
            "kept": 0,
            "packed": 0,
            "paced": 0,
            "recast": 0,
            "shared": 0,
        }
        if self.streamed:
            self.d2h_stream = self.dm.Stream(self.device)
            self.h2d_stream = self.dm.Stream(self.device)
            self._store = ActivationOffload._store_device
            self._load = ActivationOffload._load_device
            self._prefetch = ActivationOffload._fetch if prefetch else nothing
        else:
            self._store = ActivationOffload._store_host
            self._load = ActivationOffload._load_host

    def exclude(self, tensors: Iterable[Tensor]) -> None:
        """Never offload these tensors' storages (parameters, weight slots)."""
        for t in tensors:
            if t.numel():
                self.skip.add(t.untyped_storage().data_ptr())

    def rebuild(self, tensors: Iterable[Tensor]) -> None:
        """The kept storages become exactly these tensors'."""
        self.skip = set()
        self.exclude(tensors)

    @contextmanager
    def block(self, index: int):
        """Scope the hooks to block `index`; an exception drops the step's packets (their
        host buffers return to the pool)."""
        prev = self.current
        self.current = index
        self.seen = {}
        try:
            with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
                yield
        except BaseException:
            self.reset()
            raise
        finally:
            self.current = prev
            self.seen = {}

    @torch.compiler.disable
    def pack(self, t: Tensor):
        """The saved-tensors pack hook; always plain Python, never traced by Dynamo."""
        if (
            t.device != self.device
            or t.nbytes < self.min_bytes
            or t.untyped_storage().data_ptr() in self.skip
        ):
            self.stats["kept"] += 1
            return t
        param = cast_source(t)
        if param is not None:
            self.stats["recast"] += 1
            return recast_of(t, param)
        key = _key(t)
        hit = self.seen.get(key)
        if hit is not None and hit[0]() is not None:
            hit[1].users += 1
            self.stats["shared"] += 1
            return hit[1]
        self.stats["packed"] += 1
        self.stats["packed_bytes"] += t.nbytes
        pk = self._store(self, t)
        self.seen[key] = (weakref.ref(t), pk)
        self.pending.setdefault(self.current, []).append(pk)
        return pk

    @torch.compiler.disable
    def unpack(self, pk):
        """The saved-tensors unpack hook; always plain Python, never traced by Dynamo."""
        if isinstance(pk, Tensor):
            return pk
        if isinstance(pk, Recast):
            return _recast(pk)
        return self._load(self, pk)

    def _store_host(self, t: Tensor) -> Packet:
        return Packet(t.detach().clone(), self.current)

    def _load_host(self, pk: Packet) -> Tensor:
        self.pending.pop(pk.block, None)
        return pk.host

    def _take(self, t: Tensor) -> Tensor:
        """A (pinned) host byte buffer for `t`, cut from the slab; the D2H stream that
        writes it waits for the device reads of the buffers it reuses."""
        return self.slab.take(t.nbytes, self.d2h_stream)

    def _give(self, flat: Tensor, done) -> None:
        """Hand a host buffer back to the slab; `done`: its last copy's event."""
        self.slab.give(done)

    def _pace(self, kind: str, block: int, ev) -> None:
        """Record `ev` as `block`'s latest `kind` copy; while more than `inflight`
        blocks have copies queued, wait on the host for the oldest block's last one."""
        queued = self.queued[kind]
        if queued and queued[-1][0] == block:
            queued[-1][1] = ev
            return
        queued.append([block, ev])
        while len(queued) > self.inflight:
            queued.popleft()[1].synchronize()
            self.stats["paced"] += 1

    def _store_device(self, t: Tensor) -> Packet:
        flat = self._take(t)
        nbytes = t.nbytes
        host = flat[:nbytes].view(t.dtype).view(t.shape)
        self.d2h_stream.wait_stream(self.dm.current_stream(self.device))
        with self.dm.stream(self.d2h_stream):
            host.copy_(t, non_blocking=True)
        t.record_stream(self.d2h_stream)
        ev = self.dm.Event()
        ev.record(self.d2h_stream)
        self._pace("d2h", self.current, ev)
        return Packet(host, self.current, d2h=ev, flat=flat)

    def _fetch(self, block: int) -> None:
        packets = self.pending.pop(block, ())
        if packets:
            # device buffers come from the consumer (current) stream's pool
            self.h2d_stream.wait_stream(self.dm.current_stream(self.device))
        for pk in packets:
            if pk.device_copy is not None:
                continue
            copy = torch.empty(pk.host.shape, dtype=pk.host.dtype, device=self.device)
            self.h2d_stream.wait_event(pk.d2h)
            with self.dm.stream(self.h2d_stream):
                copy.copy_(pk.host, non_blocking=True)
            copy.record_stream(self.h2d_stream)
            ev = self.dm.Event()
            ev.record(self.h2d_stream)
            pk.device_copy, pk.h2d = copy, ev
            self._give(pk.flat, ev)
            pk.given = True
            self._pace("h2d", block, ev)

    def _load_device(self, pk: Packet) -> Tensor:
        """The packet's device copy (fetched now on a miss; once fetched it stays with
        the packet, freed with it); every unpack prefetches the previous block, a no-op
        once that block is fetched."""
        if pk.device_copy is None:
            if pk.given:
                raise RuntimeError(
                    "activation offload: a packet's host buffer was recycled before "
                    "its last unpack"
                )
            self.pending.setdefault(pk.block, []).append(pk)
            self._fetch(pk.block)
        self._prefetch(self, pk.block - 1)
        self.dm.current_stream(self.device).wait_event(pk.h2d)
        pk.users -= 1
        return pk.device_copy

    def reset(self) -> None:
        """Drop packets of a step that never ran backward (their buffers return)."""
        for packets in self.pending.values():
            for pk in packets:
                if pk.flat is not None and not pk.given:
                    self._give(pk.flat, pk.d2h)
        self.pending.clear()
        for queued in self.queued.values():
            queued.clear()
