"""Host storage for buffers that all come back before any is cut again (one micro-batch's
offloaded activations): cut in order from exact-size host chunks, reused from the start
once every buffer is handed back. See docs/internals/offload.md#activation-host-slab."""

from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss.offload.host.arena import HostArena
from kohakumoss_common.calls import nothing
from kohakumoss_common.units import MIB

ALIGN = 512
GROW = 64 * MIB


def _up(n: int, to: int) -> int:
    return (n + to - 1) // to * to


class HostSlab:
    """Buffers cut in order (`ALIGN`-byte aligned) from host chunks (one HostArena each,
    page-locked when `pinned`): `take(nbytes, writer)` cuts one, `give(done)` hands it
    back with the event of its last device read. When every buffer cut since the last
    rewind is back, the next `take` starts over at the first chunk, after `writer` (the
    stream writing into them) waits on every event handed back. A rewind after the cuts
    outgrew the first chunk replaces the chunks by one of the bytes cut, rounded up to
    `grow` (`sync()` first: the device has read the old ones). `nbytes`: host bytes
    held."""

    def __init__(
        self,
        device: torch.device | str,
        pinned: bool,
        sync: Callable[[], None] = nothing,
        grow: int = GROW,
    ):
        self.device = torch.device(device)
        self.pinned = pinned
        self.sync = sync
        self.grow = grow
        self.arenas: list[HostArena] = []
        self.chunks: list[Tensor] = []
        self.chunk = 0
        self.offset = 0
        self.cut = 0
        self.out = 0
        self.events: list = []
        self.grown = 0

    @property
    def nbytes(self) -> int:
        return sum(c.numel() for c in self.chunks)

    def _add(self, nbytes: int) -> None:
        arena = HostArena([((nbytes,), torch.uint8)], self.pinned, self.device)
        arena.finish()
        self.chunks.append(arena.view(0))
        self.arenas.append(arena)
        self.grown += 1

    def _drop(self) -> None:
        self.sync()
        for arena in self.arenas:
            arena.release()
        self.arenas, self.chunks = [], []

    def _rewind(self, writer) -> None:
        for ev in self.events:
            writer.wait_event(ev)
        self.events = []
        if len(self.chunks) > 1:
            total = _up(self.cut, self.grow)
            self._drop()
            self._add(total)
        self.chunk = self.offset = self.cut = 0

    def take(self, nbytes: int, writer) -> Tensor:
        """A host buffer of at least `nbytes` bytes (uint8, uninitialised)."""
        if self.out == 0 and self.cut:
            self._rewind(writer)
        n = _up(max(int(nbytes), 1), ALIGN)
        while self.chunk < len(self.chunks):
            room = self.chunks[self.chunk].numel() - self.offset
            if n <= room:
                break
            self.cut += room
            self.chunk += 1
            self.offset = 0
        if self.chunk == len(self.chunks):
            self._add(_up(n, self.grow))
        buf = self.chunks[self.chunk][self.offset : self.offset + n]
        self.offset += n
        self.cut += n
        self.out += 1
        return buf

    def give(self, done) -> None:
        """Hand a buffer back; `done`: the event its last device read recorded."""
        self.out -= 1
        self.events.append(done)

    def release(self) -> None:
        """Drop every chunk (no buffer may be out)."""
        self._drop()
        self.chunk = self.offset = self.cut = self.out = 0
        self.events = []
