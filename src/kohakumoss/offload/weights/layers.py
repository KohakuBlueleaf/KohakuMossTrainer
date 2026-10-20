"""Inference-time layer streaming for large frozen encoders: each layer's parameters
are on the device only while it runs (forward hooks, or `enter(i)` / `leave(i)` from a
forward that drives the layers itself). See docs/internals/offload.md."""

from collections.abc import Mapping, Sequence
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import module, streamed, with_index
from kohakumoss.offload.host.arena import host_copies
from kohakumoss.offload.weights.deferred import Deferred, put
from kohakumoss.offload.weights.streaming.events import SlotEvents
from kohakumoss_common.calls import nothing


def module_tensors(module: nn.Module) -> dict[str, Tensor]:
    """Parameters and buffers of `module` by name (both are streamed)."""
    out = dict(module.named_parameters())
    for name, buf in module.named_buffers():
        if buf is not None:
            out[f"buffer:{name}"] = buf
    return out


def tensor_bytes(tensors) -> int:
    """Total storage bytes of an iterable of tensors."""
    return sum(t.nbytes for t in tensors)


def module_bytes(module) -> int:
    """Parameter + buffer bytes of a module (`module_tensors`; 0 for None or a
    non-module)."""
    if not isinstance(module, nn.Module):
        return 0
    return tensor_bytes(module_tensors(module).values())


def host_copy(t: Tensor, pinned: bool, dtype: torch.dtype | None = None) -> Tensor:
    """A contiguous host copy of `t` in `dtype` (default its own), pinned when `pinned`;
    a contiguous host tensor already in `dtype` is returned as is."""
    h = t.detach().to("cpu", dtype or t.dtype).contiguous()
    return h.pin_memory() if pinned else h


def device_dtype(t: Tensor, dtypes: Mapping[int, torch.dtype] | None) -> torch.dtype:
    """The dtype module tensor `t` has on the device: its entry in `dtypes` (by tensor
    id: a host form stored in another dtype, cast by the copy), else its own."""
    return dtypes.get(id(t), t.dtype) if dtypes else t.dtype


def tensor_ids(modules: Sequence[nn.Module]) -> frozenset[int]:
    """Ids of every parameter and buffer of `modules`."""
    return frozenset(id(t) for m in modules for t in module_tensors(m).values())


def move_except(
    module: nn.Module, layers: Sequence[nn.Module], device: torch.device | str
) -> None:
    """Move every parameter/buffer of `module` not inside `layers` to `device`."""
    inside = tensor_ids(layers)
    for t in module_tensors(module).values():
        if id(t) not in inside:
            t.data = t.data.to(device)


def find_layer_list(module: nn.Module) -> nn.ModuleList:
    """The ModuleList holding most of `module`'s parameters (decoder layers)."""
    best, best_n = None, -1
    for child in module.modules():
        if isinstance(child, nn.ModuleList) and len(child) > 1:
            n = sum(p.numel() for p in child.parameters())
            if n > best_n:
                best, best_n = child, n
    if best is None:
        raise ValueError("module has no ModuleList of layers to stream")
    return best


class _Pool:
    """Ring of device slots for layers sharing one parameter signature."""

    def __init__(self, template: list[Tensor], n_slots: int):
        self.template = template
        self.n_slots = n_slots
        self.slots: list[list[Tensor]] = []
        self.held: list[int | None] = [None] * n_slots
        self.next = 0
        self.ready: list = []
        self.free: list = []

    def open(self, device, side: bool) -> None:
        self.slots = [
            [torch.empty(t.shape, dtype=t.dtype, device=device) for t in self.template]
            for _ in range(self.n_slots)
        ]
        self.held = [None] * self.n_slots
        self.next = 0
        if side:
            events = SlotEvents(module(device), device, self.n_slots)
            self.ready, self.free = events.ready, events.free

    def close(self) -> None:
        self.slots = []
        self.held = [None] * self.n_slots


class LayerStreamer:
    """Streams `layers` through `slots` device slots per layer signature.

    `host` "copy" copies the layers' tensors to one HostArena in their device dtype
    (`dtypes`), pinned when `pinned`; "file" keeps them as they are; a tensor in
    `deferred` is copied from its Deferred host form. `open()` allocates the slots
    (implicitly on the first entry), `close()` orders the compute stream after the
    last H2D copy, points every layer back at its host copy and drops the slots;
    `h2d_bytes` counts every host-to-device byte copied. The streamer holds the
    layers' tensors, not the layers (their hooks hold it)."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        device: torch.device | str,
        slots: int = 2,
        pinned: bool = True,
        hooks: bool = True,
        host: str = "copy",
        dtypes: Mapping[int, torch.dtype] | None = None,
        deferred: Mapping[int, Deferred] | None = None,
    ):
        layers = list(layers)
        deferred = deferred or {}
        self.device = with_index(device)
        self.streamed = streamed(self.device)
        self.dm = module(self.device)
        self.host: list[list[Tensor]] = []
        self.sources: list[list] = []
        self.targets: list[list[Tensor]] = []
        self.pool_of: list[_Pool] = []
        self.layer_bytes: list[int] = []
        pools: dict[tuple, _Pool] = {}
        per_layer = [module_tensors(layer) for layer in layers]
        copied = [
            t
            for params in per_layer
            for t in params.values()
            if host == "copy" and id(t) not in deferred
        ]
        wants_copied = [device_dtype(t, dtypes) for t in copied]
        self.arena, _ = host_copies(
            copied, pinned and self.streamed, self.device, wants_copied
        )
        for params in per_layer:
            hosts, wants, sources = [], [], []
            for t in params.values():
                want = device_dtype(t, dtypes)
                hosts.append(t.data)
                wants.append(want)
                sources.append(deferred.get(id(t), t.data))
            shapes = [tuple(h.shape) for h in hosts]
            sig = tuple(zip(params, shapes, wants))
            if sig not in pools:
                template = [
                    torch.empty(s, dtype=w, device="meta")
                    for s, w in zip(shapes, wants)
                ]
                pools[sig] = _Pool(template, slots)
            self.pool_of.append(pools[sig])
            self.host.append(hosts)
            self.sources.append(sources)
            self.targets.append(list(params.values()))
            self.layer_bytes.append(tensor_bytes(sources))
        self.pools = list(pools.values())
        self.slot_of: dict[int, int] = {}
        self.is_open = False
        self.h2d_bytes = 0
        if self.streamed:
            self.stream = self.dm.Stream(self.device)
            self._copy, self._wait, self._free = (
                LayerStreamer._copy_device,
                LayerStreamer._wait_device,
                LayerStreamer._free_device,
            )
        else:
            self._copy, self._wait, self._free = (
                LayerStreamer._copy_host,
                nothing,
                nothing,
            )
        n = len(layers)
        self.ahead = [
            partial(LayerStreamer._prefetch, i=i + 1) if i + 1 < n else nothing
            for i in range(n)
        ]
        self.handles = []
        if hooks:
            for i, layer in enumerate(layers):
                pre, post = partial(self._pre, i), partial(self._post, i)
                self.handles.append(layer.register_forward_pre_hook(pre))
                self.handles.append(layer.register_forward_hook(post))

    def slot_bytes(self) -> int:
        """Device bytes the open ring holds."""
        return sum(p.n_slots * tensor_bytes(p.template) for p in self.pools)

    def open(self) -> None:
        for pool in self.pools:
            pool.open(self.device, self.streamed)
        self.slot_of.clear()
        self.is_open = True

    def close(self) -> None:
        if not self.is_open:
            return
        if self.streamed:
            self.dm.current_stream(self.device).wait_stream(self.stream)
        for i, targets in enumerate(self.targets):
            for t, h in zip(targets, self.host[i]):
                t.data = h
        for pool in self.pools:
            pool.close()
        self.slot_of.clear()
        self.is_open = False

    def _load(self, i: int) -> int:
        pool = self.pool_of[i]
        slot = pool.next
        pool.next = (slot + 1) % len(pool.slots)
        old = pool.held[slot]
        if old is not None:
            self.slot_of.pop(old, None)
        pool.held[slot] = i
        self.slot_of[i] = slot
        self._copy(self, pool, slot, self.sources[i])
        self.h2d_bytes += self.layer_bytes[i]
        return slot

    def _copy_host(self, pool: _Pool, slot: int, src: list) -> None:
        for dst, s in zip(pool.slots[slot], src):
            put(dst, s, self.device, False)

    def _copy_device(self, pool: _Pool, slot: int, src: list) -> None:
        self.stream.wait_event(pool.free[slot])
        with self.dm.stream(self.stream):
            for dst, s in zip(pool.slots[slot], src):
                put(dst, s, self.device, True)
        pool.ready[slot].record(self.stream)

    def _wait_device(self, pool: _Pool, slot: int) -> None:
        self.dm.current_stream(self.device).wait_event(pool.ready[slot])

    def _free_device(self, pool: _Pool, slot: int) -> None:
        pool.free[slot].record(self.dm.current_stream(self.device))

    def _prefetch(self, i: int) -> None:
        if i not in self.slot_of:
            self._load(i)

    def enter(self, i: int) -> None:
        """Layer i's tensors point at its device slot (loaded if not prefetched)."""
        if not self.is_open:
            self.open()
        slot = self.slot_of.get(i)
        if slot is None:
            slot = self._load(i)
        pool = self.pool_of[i]
        self._wait(self, pool, slot)
        for t, s in zip(self.targets[i], pool.slots[slot]):
            t.data = s
        self.ahead[i](self)

    def leave(self, i: int) -> None:
        """Layer i's slot is released to the ring; its tensors point at the host copy."""
        pool = self.pool_of[i]
        self._free(self, pool, self.slot_of[i])
        for t, h in zip(self.targets[i], self.host[i]):
            t.data = h

    def _pre(self, i: int, module, args) -> None:
        self.enter(i)

    def _post(self, i: int, module, args, output) -> None:
        self.leave(i)

    def remove(self) -> None:
        """Drop the hooks and the host copies (the caller re-points the layers'
        tensors)."""
        for h in self.handles:
            h.remove()
        self.handles = []
        self.host = []
        self.sources = []
        self.arena.release()
        self.targets = []
