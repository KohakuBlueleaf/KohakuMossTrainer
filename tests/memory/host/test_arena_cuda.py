"""HostArena on CUDA: one exact-size registered buffer where per-tensor pinning holds
each tensor's power-of-two class; an H2D from an arena view is asynchronous (the host
returns before the copy ends, and the copy overlaps a kernel on another stream) where a
pageable buffer blocks; a host write into a region the device still reads corrupts the
copy unless it waits on the copy's event; a refused registration's error is reset so
the next kernel launch does not raise it; a frozen WeightStream keeps its blocks in one
registered arena."""

import sys
import time
from pathlib import Path

import pytest
import torch

from kohakumoss.model.linear import MossLinear
from kohakumoss.offload.host import arena as ha
from kohakumoss.offload.host import cudart
from kohakumoss.offload.weights.stream import WeightStream
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
MIB = 2**20
BIG = 256 * MIB


def _arena_view(nbytes: int = BIG) -> tuple[ha.HostArena, torch.Tensor]:
    arena = ha.HostArena([((nbytes // 4,), torch.float32)], True, "cuda")
    arena.fill(0, torch.ones(nbytes // 4))
    return arena.finish(), arena.view(0)


def page_locked(t: torch.Tensor) -> bool:
    """The driver holds `t`'s range registered already (a second registration of it
    is refused; the refusal's error is reset)."""
    if cudart.register(t.data_ptr(), t.nbytes) is None:
        cudart.unregister(t.data_ptr())
        return False
    cudart.reset_last_error()
    return True


class Busy:
    """About `ms` of fp16 matmuls, queued on a stream by `queue(stream)`."""

    def __init__(self, ms: float):
        self.a = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
        self.a @ self.a
        self.reps = 8
        per = _timed(lambda: self.queue(torch.cuda.current_stream())) / 8
        self.reps = max(1, int(ms / 1e3 / max(per, 1e-5)))

    def queue(self, stream) -> None:
        with torch.cuda.stream(stream):
            for _ in range(self.reps):
                self.a @ self.a


def _copy(stream, dev, host) -> None:
    with torch.cuda.stream(stream):
        dev.copy_(host, non_blocking=True)


def _timed(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def assert_async(host: torch.Tensor) -> None:
    dev = torch.empty_like(host, device="cuda")
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _copy(side, dev, host)
    returned = time.perf_counter() - start
    done = torch.cuda.Event()
    done.record(side)
    pending = not done.query()
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    assert pending and returned < 0.25 * total, (returned, total)
    busy = Busy(total * 1e3)
    main = torch.cuda.current_stream()
    alone_k = _timed(lambda: busy.queue(main))
    alone_c = _timed(lambda: _copy(side, dev, host))
    both = _timed(lambda: (busy.queue(main), _copy(side, dev, host)))
    overlap = alone_k + alone_c - both
    assert overlap > 0.5 * min(alone_k, alone_c), (both, alone_k, alone_c)


def test_arena_copies_are_asynchronous():
    arena, view = _arena_view()
    assert_async(view)
    pageable = torch.ones(BIG // 4)
    must_fail(lambda: assert_async(pageable), "a pageable buffer")


def run_race(view: torch.Tensor, wait: bool) -> torch.Tensor:
    """Copy `view` (all ones) to the device behind a queued kernel, then write twos
    into it on the host, waiting on the copy's event first when `wait`."""
    view.fill_(1.0)
    dev = torch.empty_like(view, device="cuda")
    side = torch.cuda.Stream()
    Busy(100.0).queue(side)
    with torch.cuda.stream(side):
        dev.copy_(view, non_blocking=True)
    read = torch.cuda.Event()
    read.record(side)
    if wait:
        read.synchronize()
    view.fill_(2.0)
    torch.cuda.synchronize()
    return dev


def assert_no_race(view, wait: bool) -> None:
    untouched = bool((run_race(view, wait) == 1.0).all())
    assert untouched, "the device read the host's later write"


def test_a_write_waits_for_the_copy_reading_the_region():
    arena, view = _arena_view()
    assert_no_race(view, wait=True)
    must_fail(lambda: assert_no_race(view, wait=False), "a write without the wait")


def check_reset(reset) -> None:
    t = torch.empty(4 * MIB, dtype=torch.uint8)
    assert cudart.register(t.data_ptr(), t.nbytes) is None
    try:
        assert cudart.register(t.data_ptr(), t.nbytes) is not None
        reset()
        try:
            torch.ones(4, device="cuda").add_(1)
        except torch.AcceleratorError as err:
            raise AssertionError(f"a stale runtime error raised: {err}") from err
    finally:
        cudart.unregister(t.data_ptr())


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc maps")
def test_the_reset_library_is_torchs_loaded_runtime():
    lib = cudart.runtime_library()
    mapped = cudart.mapped_libraries(cudart.MAPS.read_text(), "libcudart.so")
    assert lib is not None and lib._name in mapped, (lib, mapped)


def test_refused_registration_error_is_reset():
    check_reset(cudart.reset_last_error)
    must_fail(lambda: check_reset(lambda: None), "the error left set")


def _shmem() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssShmem:"):
            return int(line.split()[1]) * 1024
    return 0


def _pinned_reserved(make) -> int:
    """Host bytes `make()`'s pinned allocations hold (cudaHostAlloc pages are shared
    memory on Linux)."""
    torch.cuda.synchronize()
    torch.accelerator.empty_host_cache()
    before = _shmem()
    held = make()
    for t in held:
        t.fill_(1)
    after = _shmem()
    del held
    torch.accelerator.empty_host_cache()
    return after - before


def assert_exact(make_arena, specs) -> None:
    payload = sum(ha.aligned(ha.spec_bytes(s)) for s in specs)
    arena = make_arena(specs)
    assert arena.pinned and all(page_locked(arena.view(i)) for i in range(len(specs)))
    assert arena.nbytes <= ha.aligned(payload, ha.PAGE), (arena.nbytes, payload)


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc shared memory")
def test_cuda_arena_is_exact_where_pinning_rounds():
    specs = [((3 * MIB + 7,), torch.uint8), ((1536, 4096), torch.float16)] * 3
    arena = ha.HostArena(specs, True, "cuda")
    for i, (shape, dtype) in enumerate(specs):
        arena.fill(i, torch.ones(shape, dtype=dtype))
    arena.finish()
    assert_exact(lambda s: arena, specs)
    assert not page_locked(torch.empty(1 << 20, dtype=torch.uint8))
    rounded = _pinned_reserved(
        lambda: [torch.empty(s, dtype=d, pin_memory=True) for s, d in specs]
    )
    payload = sum(ha.spec_bytes(s) for s in specs)
    assert rounded > payload * 1.2, (rounded, payload)

    def per_tensor_arena(s):
        a = ha.HostArena(s, True, "cuda")
        a.nbytes = sum(ha.pow2(ha.spec_bytes(x)) for x in s)
        return a.finish()

    must_fail(lambda: assert_exact(per_tensor_arena, specs), "power-of-two classes")


def _frozen_stream() -> tuple[WeightStream, list[torch.Tensor]]:
    blocks, want = [], []
    for _ in range(3):
        block = torch.nn.Module()
        block.lin = MossLinear(320, 192, bias=False)
        torch.nn.init.normal_(block.lin.weight)
        want.append(block.lin.weight.detach().to(torch.float16))
        blocks.append(block)
    stream = WeightStream(blocks, "cuda", slots=2, dtype=torch.float16)
    return stream, want


def assert_in_arena(hosts, arena, want) -> None:
    assert arena is not None and arena.kind == "registered"
    assert all(page_locked(h) for h in hosts)
    assert all(torch.equal(h, w) for h, w in zip(hosts, want))
    base = arena.buffers[0].data_ptr()
    assert all(base <= h.data_ptr() < base + arena.nbytes for h in hosts)


def test_frozen_stream_blocks_live_in_one_registered_arena():
    stream, want = _frozen_stream()
    hosts = [stream.host[i]["lin"] for i in range(3)]
    assert_in_arena(hosts, stream.host_arena, want)
    copies = [h.clone().pin_memory() for h in hosts]
    must_fail(
        lambda: assert_in_arena(copies, stream.host_arena, want), "per-tensor pinning"
    )
