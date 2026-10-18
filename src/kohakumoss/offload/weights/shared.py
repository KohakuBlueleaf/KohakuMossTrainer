"""Host memory the ranks of one node share (DDP over streamed `swap` blocks): one
/dev/shm region rank 0 creates and fills and every rank maps, and the block owner plan
of the host optimiser shards. Linux only, one node. See docs/guides/multi-gpu.md (shared
host masters)."""

import math
import os
import uuid

import torch
import torch.distributed as dist
from torch import Tensor

ALIGN = 64
SHM_DIR = "/dev/shm"


def block_owners(n: int, world: int) -> list[int]:
    """The rank that holds the optimiser state of, and steps, each of `n` blocks."""
    return [i % world for i in range(n)]


def owned(owners: list[int], rank: int) -> list[int]:
    """The blocks `rank` owns."""
    return [i for i, r in enumerate(owners) if r == rank]


def aligned(nbytes: int) -> int:
    """`nbytes` rounded up to the region's view alignment."""
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


def nbytes_of(shape, dtype: torch.dtype) -> int:
    """Bytes of a dense tensor of `shape` in `dtype`."""
    return math.prod(shape) * dtype.itemsize


class SharedHost:
    """`nbytes` of host memory mapped by every rank of the default process group.
    `take(shape, dtype)` hands out consecutive aligned views in call order (the same on
    every rank); `writer` (rank 0) fills them, every rank calls `ready()` before reading
    them. `pin` page-locks the mapping for asynchronous copies (CUDA)."""

    def __init__(self, nbytes: int, pin: bool = False, group=None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.writer = self.rank == 0
        self.nbytes = max(int(nbytes), 1)
        name = [f"kohakumoss-{uuid.uuid4().hex}" if self.writer else None]
        dist.broadcast_object_list(name, src=0, group=group)
        path = os.path.join(SHM_DIR, name[0])
        if self.writer:
            with open(path, "wb") as f:
                f.truncate(self.nbytes)
        dist.barrier(group=group)
        self.buf = torch.from_file(
            path, shared=True, size=self.nbytes, dtype=torch.uint8
        )
        dist.barrier(group=group)
        if self.writer:
            os.unlink(path)
        self.pinned = pin and self._register()
        self.offset = 0

    def _register(self) -> bool:
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostRegister(self.buf.data_ptr(), self.nbytes, 0)
        return err == cudart.cudaError.success

    def take(self, shape, dtype: torch.dtype) -> Tensor:
        """The next view of `shape` / `dtype` in the region."""
        n = nbytes_of(shape, dtype)
        start = self.offset
        if start + n > self.nbytes:
            raise ValueError(f"shared host region of {self.nbytes} B is full")
        self.offset = aligned(start + n)
        return self.buf[start : start + n].view(dtype).view(shape)

    def ready(self) -> None:
        """Every rank: the writer's values (or updates) are in place."""
        dist.barrier(group=self.group)

    def close(self) -> None:
        if self.pinned:
            torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
            self.pinned = False
