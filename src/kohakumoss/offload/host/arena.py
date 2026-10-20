"""Exact-size host storage for the many tensors one offload engine keeps on the host.

One pin_memory allocation per tensor holds the next power of two of its bytes (the
caching host allocator's size classes). A HostArena lays every tensor of an engine out
in one buffer instead: on CUDA a pageable buffer page-locked once it is filled
(cudaHostRegister), elsewhere power-of-two pinned chunks packed first-fit, pageable
tensors one allocation each. Views are filled one at a time, so filling from host
tensors that are freed as they go peaks at one tensor over the arena. See
docs/internals/offload.md#host-arena."""

import math
import weakref
from collections.abc import Sequence

import torch
from torch import Tensor

from kohakumoss.logging.log import get_logger
from kohakumoss.offload.accel import streamed
from kohakumoss.offload.host import cudart
from kohakumoss_common.hostmem import trim_heap
from kohakumoss_common.units import GIB, MIB

log = get_logger(__name__)

ALIGN = 64
PAGE = 4096
CHUNK = 256 * MIB
Spec = tuple[torch.Size | tuple[int, ...], torch.dtype]


def aligned(nbytes: int, align: int = ALIGN) -> int:
    return (nbytes + align - 1) // align * align


def pow2(nbytes: int) -> int:
    """The caching host allocator's size class of a pinned request of `nbytes`."""
    return 1 << max(int(nbytes) - 1, 0).bit_length()


def spec_bytes(spec: Spec) -> int:
    shape, dtype = spec
    return math.prod(shape) * dtype.itemsize


def pack(sizes: Sequence[int], chunk: int) -> tuple[list[tuple[int, int]], list[int]]:
    """First-fit decreasing: ([(chunk index, offset)] per size, [chunk bytes]); a size
    over `chunk` gets a chunk of its own power of two."""
    where: list[tuple[int, int] | None] = [None] * len(sizes)
    caps: list[int] = []
    used: list[int] = []
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        n = aligned(sizes[i])
        for c, cap in enumerate(caps):
            if used[c] + n <= cap:
                where[i] = (c, used[c])
                used[c] += n
                break
        else:
            caps.append(max(chunk, pow2(n)))
            used.append(n)
            where[i] = (len(caps) - 1, 0)
    return where, caps


def arena_kind(pinned: bool, device: torch.device | str) -> str:
    """How an arena of `device`'s copies stores its tensors: "registered" (CUDA, one
    buffer page-locked after filling), "chunks" (other pinned devices) or "pageable"."""
    device = torch.device(device)
    if not (pinned and streamed(device)):
        return "pageable"
    return "registered" if device.type == "cuda" else "chunks"


def chunk_size(sizes: Sequence[int], chunk: int = CHUNK) -> int:
    return min(chunk, pow2(max(sum(aligned(n) for n in sizes), 1)))


def arena_bytes(sizes: Sequence[int], kind: str, chunk: int = CHUNK) -> int:
    """Host bytes an arena of tensors of `sizes` bytes holds under `kind`."""
    match kind:
        case "pageable":
            return sum(sizes)
        case "registered":
            return aligned(max(sum(aligned(n) for n in sizes), 1), PAGE)
    return sum(pack(sizes, chunk_size(sizes, chunk))[1])


def pinned_empty(nbytes: int) -> Tensor:
    """A pinned uint8 host buffer from the caching host allocator."""
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def _unregister(buf: Tensor, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    cudart.unregister(buf.data_ptr())


class HostArena:
    """Host storage for `specs` (shape, dtype) on behalf of `device`'s copies: `view(i)`
    is spec i's tensor, `finish()` page-locks a CUDA arena once its views are filled.
    `pinned` (and a device with pinned memory) places them in page-locked storage, else
    each is a plain host tensor; a refused registration moves the views into pinned
    chunks (`kind` "chunks", a WARNING), re-pointing every view handed out and every
    tensor `track`ed onto one. `nbytes`: host bytes held; `payload`: the specs'."""

    def __init__(
        self,
        specs: Sequence[Spec],
        pinned: bool,
        device: torch.device | str = "cpu",
        chunk: int = CHUNK,
    ):
        self.specs = [(torch.Size(s), d) for s, d in specs]
        self.sizes = [spec_bytes(s) for s in self.specs]
        self.payload = sum(self.sizes)
        self.device = torch.device(device)
        self.chunk = chunk
        self.kind = arena_kind(pinned, self.device)
        self.pinned = self.kind != "pageable"
        self.views: list[Tensor | None] = [None] * len(self.specs)
        self.tracked: list[list[Tensor]] = [[] for _ in self.specs]
        self.buffers: list[Tensor | None] = []
        self.registered = None
        self.nbytes = arena_bytes(self.sizes, self.kind, chunk)
        match self.kind:
            case "registered":
                self.offsets, total = [], 0
                for n in self.sizes:
                    self.offsets.append(total)
                    total += aligned(n)
                raw = torch.empty(self.nbytes + PAGE, dtype=torch.uint8)
                skip = (-raw.data_ptr()) % PAGE
                self.buffers = [raw[skip : skip + self.nbytes]]
            case "chunks":
                self._chunks()

    def _chunks(self) -> None:
        self.kind = "chunks"
        self.where, self.caps = pack(self.sizes, chunk_size(self.sizes, self.chunk))
        self.buffers = [None] * len(self.caps)
        self.nbytes = sum(self.caps)

    def __len__(self) -> int:
        return len(self.specs)

    def _buffer(self, c: int) -> Tensor:
        buf = self.buffers[c]
        if buf is None:
            buf = pinned_empty(self.caps[c])
            self.buffers[c] = buf
        return buf

    def track(self, i: int, t: Tensor) -> None:
        """`t` shares spec i's view storage (a parameter or a re-pointed tensor over
        it): a move to pinned chunks re-points it too."""
        self.tracked[i].append(t)

    def _make(self, i: int) -> Tensor:
        shape, dtype = self.specs[i]
        n = self.sizes[i]
        match self.kind:
            case "pageable":
                return torch.empty(shape, dtype=dtype)
            case "registered":
                start = self.offsets[i]
                return self.buffers[0][start : start + n].view(dtype).view(shape)
        c, start = self.where[i]
        return self._buffer(c)[start : start + n].view(dtype).view(shape)

    def view(self, i: int) -> Tensor:
        """Spec i's host tensor (uninitialised until written)."""
        v = self.views[i]
        if v is None:
            v = self.views[i] = self._make(i)
        return v

    def fill(self, i: int, src: Tensor) -> Tensor:
        """Spec i's tensor holding `src`'s values (cast to the spec dtype)."""
        return self.view(i).copy_(src)

    def finish(self) -> "HostArena":
        """Page-lock a CUDA arena once its views are filled; a refused registration
        clears the runtime's error and moves the arena into pinned chunks."""
        if self.kind != "registered" or self.registered is not None or not self:
            return self
        buf = self.buffers[0]
        err = cudart.register(buf.data_ptr(), buf.numel())
        if err is None:
            self.registered = weakref.finalize(self, _unregister, buf, self.device)
            return self
        cudart.reset_last_error()
        held = self.nbytes
        self._to_chunks()
        log.warning(
            f"cudaHostRegister refused {held / GIB:.2f} GiB ({err}): pinned chunks "
            f"instead, {self.nbytes / GIB:.2f} GiB held"
        )
        return self

    def _to_chunks(self) -> None:
        """Every view handed out, and every tensor tracked onto one, re-pointed at the
        same values in pinned chunks."""
        old = list(self.views)
        self._chunks()
        self.views = [None] * len(self.specs)
        for i, v in enumerate(old):
            if v is None:
                continue
            new = self._make(i).copy_(v)
            for t in (v, *self.tracked[i]):
                t.data = new
            self.views[i] = v

    def release(self) -> None:
        """Drop the storage once the device has finished with it (views taken earlier
        keep their memory until dropped too)."""
        if self.registered is not None:
            self.registered()
            self.registered = None
        self.views = [None] * len(self.specs)
        self.tracked = [[] for _ in self.specs]
        self.buffers = []


def host_tensors(
    specs: Sequence[Spec],
    pinned: bool,
    device: torch.device | str,
    fill: Sequence[Tensor | None] | None = None,
) -> tuple[HostArena, list[Tensor]]:
    """(arena, a host tensor per spec), each holding its `fill` entry's values (cast to
    the spec dtype), zeros where it is None or `fill` is not given; page-locked once
    filled."""
    arena = HostArena(specs, pinned, device)
    fill = list(fill) if fill is not None else [None] * len(arena)
    for i, src in enumerate(fill):
        view = arena.view(i)
        if src is None:
            view.zero_()
        else:
            view.copy_(src)
    arena.finish()
    return arena, [arena.view(i) for i in range(len(arena))]


def host_copies(
    tensors: Sequence[Tensor],
    pinned: bool,
    device: torch.device | str,
    dtypes: Sequence[torch.dtype] | None = None,
) -> tuple[HostArena, list[Tensor]]:
    """(arena, host copies of `tensors`, each in its `dtypes` entry or its own dtype),
    each tensor re-pointed at its copy as the copy is made, so a host source nothing
    else holds is freed while the arena fills (the C heap's freed pages returned
    after)."""
    dtypes = list(dtypes) if dtypes is not None else [t.dtype for t in tensors]
    arena = HostArena([(t.shape, d) for t, d in zip(tensors, dtypes)], pinned, device)
    for i, t in enumerate(tensors):
        t.data = arena.fill(i, t.detach())
        arena.track(i, t)
    arena.finish()
    trim_heap()
    return arena, [arena.view(i) for i in range(len(arena))]
