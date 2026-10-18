"""The device slot ring of a weight stream: which block each slot holds and which slot
a load overwrites. See docs/internals/offload.md#weight-streaming."""

import torch
from torch import Tensor

from kohakumoss.offload.weights.streaming.copier import H2DCopier


class SlotRing:
    """`n_slots` device slots, each one block's tensors (`layout`: key -> (shape,
    dtype)); a load takes the least recently used slot not holding the active block or
    one of the first `keep` blocks (kept once loaded). `copier` fills a slot from
    `host[i]` and orders it against the compute stream."""

    def __init__(
        self,
        host: list[dict[str, Tensor]],
        layout: dict[str, tuple[torch.Size, torch.dtype]],
        device: torch.device,
        n_slots: int,
        keep: int,
        copier: H2DCopier,
        stats: dict,
    ):
        self.host = host
        self.n = len(host)
        self.n_slots = n_slots
        self.keep = keep
        self.copier = copier
        self.stats = stats
        self.slots = [
            {k: torch.empty(s, dtype=d, device=device) for k, (s, d) in layout.items()}
            for _ in range(n_slots)
        ]
        self.slot_of: dict[int, int] = {}
        self.held: list[int | None] = [None] * n_slots
        self.age = [0] * n_slots
        self.clock = 0
        self.active: int | None = None

    def slot(self, block: int, key: str) -> Tensor:
        """The device copy of (block, key), loading it when it is not resident."""
        slot = self.slot_of.get(block)
        if slot is None:
            self.ensure(block)
            slot = self.slot_of[block]
        return self.slots[slot][key]

    def _victim(self) -> int:
        """The oldest slot not holding the active block or a kept block."""
        candidates = [
            s
            for s in range(self.n_slots)
            if self.held[s] is None
            or (self.held[s] != self.active and self.held[s] >= self.keep)
        ]
        return min(candidates, key=lambda s: self.age[s])

    def _load(self, index: int, slot: int) -> None:
        old = self.held[slot]
        if old is not None:
            self.slot_of.pop(old, None)
        self.held[slot] = index
        self.slot_of[index] = slot
        self.clock += 1
        self.age[slot] = self.clock
        self.stats["loads"] += 1
        self.copier.copy(self.host[index], self.slots[slot], slot, index)

    def ensure(self, index: int) -> None:
        """Block `index` is resident and the compute stream waits for its copy."""
        slot = self.slot_of.get(index)
        if slot is None:
            slot = self._victim()
            self._load(index, slot)
        self.active = index
        self.clock += 1
        self.age[slot] = self.clock
        self.copier.wait(slot)

    def prefetch(self, index: int) -> None:
        if 0 <= index < self.n and index not in self.slot_of:
            self._load(index, self._victim())

    def release(self, index: int) -> None:
        """The compute stream is done with block `index`; its slot may be overwritten."""
        slot = self.slot_of.get(index)
        if slot is not None:
            self.copier.mark_free(slot)

    def forget(self) -> None:
        """No slot holds a block: the next use of every block reloads it."""
        self.slot_of.clear()
        self.held = [None] * self.n_slots

    def storages(self) -> list[Tensor]:
        """Ring slot tensors (to exclude from activation offload)."""
        return [t for slot in self.slots for t in slot.values()]
