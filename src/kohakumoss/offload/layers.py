"""Inference-time layer streaming for large frozen encoders (the 9B text encoder on
<=16 GB cards): each layer's parameters are on the device only while it runs.

Forward pre-hooks swap layer i's parameter storage to a device ring slot (prefetching
layer i+1 on a side stream); forward hooks swap it back to the host copy. Layers of
different structure (e.g. linear- vs full-attention layers) get separate slot pools.
Per-layer tensor lists, the copy path and the prefetch target are bound at build.
"""

from collections.abc import Sequence
from functools import partial

import torch
from torch import Tensor, nn


def _tensors(module: nn.Module) -> dict[str, Tensor]:
    """Parameters and buffers of `module` by name (both are streamed)."""
    out = dict(module.named_parameters())
    for name, buf in module.named_buffers():
        if buf is not None:
            out[f"buffer:{name}"] = buf
    return out


def move_except(
    module: nn.Module, layers: Sequence[nn.Module], device: torch.device | str
) -> None:
    """Move every parameter/buffer of `module` not inside `layers` to `device`."""
    inside = {id(t) for layer in layers for t in _tensors(layer).values()}
    for t in _tensors(module).values():
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

    def __init__(self, template: list[Tensor], resident: int, device, cuda: bool):
        self.slots = [
            [torch.empty(t.shape, dtype=t.dtype, device=device) for t in template]
            for _ in range(resident)
        ]
        self.held: list[int | None] = [None] * resident
        self.next = 0
        if cuda:
            cur = torch.cuda.current_stream(device)
            self.ready = [torch.cuda.Event() for _ in range(resident)]
            self.free = [torch.cuda.Event() for _ in range(resident)]
            for ev in self.free:
                ev.record(cur)


def _no_prefetch() -> None:
    return None


class LayerStreamer:
    """Streams `layers` through `resident` device slots per layer signature."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        device: torch.device | str,
        resident: int = 2,
        pinned: bool = True,
    ):
        self.layers = list(layers)
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.host: list[list[Tensor]] = []
        self.targets: list[list[Tensor]] = []
        self.pool_of: list[_Pool] = []
        pools: dict[tuple, _Pool] = {}
        for layer in self.layers:
            params = _tensors(layer)
            hosts = []
            for t in params.values():
                h = t.detach().to("cpu").contiguous()
                if pinned and self.cuda:
                    h = h.pin_memory()
                t.data = h
                hosts.append(h)
            sig = tuple((n, tuple(t.shape), t.dtype) for n, t in zip(params, hosts))
            if sig not in pools:
                pools[sig] = _Pool(hosts, resident, self.device, self.cuda)
            self.pool_of.append(pools[sig])
            self.host.append(hosts)
            self.targets.append(list(params.values()))
        self.slot_of: dict[int, int] = {}
        if self.cuda:
            self.stream = torch.cuda.Stream(self.device)
            self._copy, self._wait, self._free = (
                self._copy_device,
                self._wait_device,
                self._free_device,
            )
        else:
            self._copy, self._wait, self._free = (
                self._copy_host,
                self._nothing,
                self._nothing,
            )
        n = len(self.layers)
        self.handles = []
        for i, layer in enumerate(self.layers):
            ahead = partial(self._prefetch, i + 1) if i + 1 < n else _no_prefetch
            pre = partial(self._pre, i, ahead)
            post = partial(self._post, i)
            self.handles.append(layer.register_forward_pre_hook(pre))
            self.handles.append(layer.register_forward_hook(post))

    def _load(self, i: int) -> int:
        pool = self.pool_of[i]
        slot = pool.next
        pool.next = (slot + 1) % len(pool.slots)
        old = pool.held[slot]
        if old is not None:
            self.slot_of.pop(old, None)
        pool.held[slot] = i
        self.slot_of[i] = slot
        self._copy(pool, slot, self.host[i])
        return slot

    def _copy_host(self, pool: _Pool, slot: int, src: list[Tensor]) -> None:
        for dst, s in zip(pool.slots[slot], src):
            dst.copy_(s)

    def _copy_device(self, pool: _Pool, slot: int, src: list[Tensor]) -> None:
        self.stream.wait_event(pool.free[slot])
        with torch.cuda.stream(self.stream):
            for dst, s in zip(pool.slots[slot], src):
                dst.copy_(s, non_blocking=True)
        pool.ready[slot].record(self.stream)

    def _nothing(self, pool: _Pool, slot: int) -> None:
        return None

    def _wait_device(self, pool: _Pool, slot: int) -> None:
        torch.cuda.current_stream(self.device).wait_event(pool.ready[slot])

    def _free_device(self, pool: _Pool, slot: int) -> None:
        pool.free[slot].record(torch.cuda.current_stream(self.device))

    def _prefetch(self, i: int) -> None:
        if i not in self.slot_of:
            self._load(i)

    def _pre(self, i: int, ahead, module, args) -> None:
        slot = self.slot_of.get(i)
        if slot is None:
            slot = self._load(i)
        pool = self.pool_of[i]
        self._wait(pool, slot)
        for t, s in zip(self.targets[i], pool.slots[slot]):
            t.data = s
        ahead()

    def _post(self, i: int, module, args, output) -> None:
        pool = self.pool_of[i]
        self._free(pool, self.slot_of[i])
        for t, h in zip(self.targets[i], self.host[i]):
            t.data = h

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
