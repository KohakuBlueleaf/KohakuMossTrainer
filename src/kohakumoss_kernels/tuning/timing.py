"""Device-time measurement: busy warmup, L2 flush between calls, CUDA-graph replay,
back-to-back bursts queued behind a device spin."""

import time
from collections.abc import Callable
from functools import cache

import torch

from kohakumoss_kernels.hardware.current import on_device, synchronize, with_index

WARMUP_MS = 25.0
CAPTURE = "thread_local"
SPIN_PROBE = 1 << 22
# The spin covers this multiple of the burst's measured host launch time.
SPIN_MARGIN = 2.0


def _elapsed_ms(fn: Callable[[], object], calls: int, device: torch.device) -> float:
    """Device ms of `calls` calls (wall ms on CPU, which has no events)."""
    if device.type == "cpu":
        t = time.perf_counter()
        for _ in range(calls):
            fn()
        return (time.perf_counter() - t) * 1e3
    with on_device(device):
        start = torch.Event(device=device, enable_timing=True)
        end = torch.Event(device=device, enable_timing=True)
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)


def warm(fn: Callable[[], object], device: torch.device, warmup_ms: float = WARMUP_MS):
    """Run `fn` until `warmup_ms` of device time has passed (boost clocks reached)."""
    calls, spent = 2, 0.0
    while spent < warmup_ms:
        spent += _elapsed_ms(fn, calls, device)
        calls *= 2


def best_ms(
    fn: Callable[[], object],
    device: torch.device,
    iters: int = 10,
    flush: torch.Tensor | None = None,
) -> float:
    """Best-of-`iters` device time of `fn` in ms; `flush` is zeroed before each call."""
    warm(fn, device)
    best = float("inf")
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        best = min(best, _elapsed_ms(fn, 1, device))
    return best


@cache
def spin_cycles_per_ms(index: int) -> float:
    """Clock cycles `torch.cuda._sleep` spins per device ms on CUDA device `index`."""
    device = torch.device("cuda", index)
    with on_device(device):
        torch.cuda._sleep(SPIN_PROBE)
        start = torch.Event(device=device, enable_timing=True)
        end = torch.Event(device=device, enable_timing=True)
        start.record()
        torch.cuda._sleep(SPIN_PROBE)
        end.record()
        end.synchronize()
        return SPIN_PROBE / max(start.elapsed_time(end), 1e-3)


def burst_ms(
    fn: Callable[[], object], device: torch.device, calls: int = 8, iters: int = 5
) -> float:
    """Best-of-`iters` device ms per call of `calls` back-to-back calls, issued while
    the device spins so that all of them are queued before the first starts: the host's
    launch time is not counted (plain event timing off CUDA)."""
    if device.type != "cuda":
        return best_ms(fn, device, iters)
    warm(fn, device)
    began = time.perf_counter()
    for _ in range(calls):
        fn()
    host_ms = (time.perf_counter() - began) * 1e3
    synchronize(device)
    cycles = int(spin_cycles_per_ms(with_index(device).index) * SPIN_MARGIN * host_ms)
    best = float("inf")
    with on_device(device):
        for _ in range(iters):
            start = torch.Event(device=device, enable_timing=True)
            end = torch.Event(device=device, enable_timing=True)
            torch.cuda._sleep(cycles)
            start.record()
            for _ in range(calls):
                fn()
            end.record()
            end.synchronize()
            best = min(best, start.elapsed_time(end) / calls)
    return best


def l2_flush(device: torch.device, l2_bytes: int) -> torch.Tensor:
    """A buffer larger than L2; zeroing it evicts the previous call's operands."""
    return torch.empty(
        max(2 * l2_bytes, 8 << 20) // 4, device=device, dtype=torch.int32
    )


def graph_ms(
    fn: Callable[[], object], device: torch.device, calls: int = 5, iters: int = 10
) -> float:
    """Device ms per call replayed from a thread-local CUDA graph of `calls` launches
    (device time only), freed after the measurement; plain timing when capture fails
    or off CUDA."""
    fn()
    synchronize(device)
    if device.type != "cuda":
        return best_ms(fn, device, iters)
    graph = torch.cuda.CUDAGraph()
    try:
        with on_device(device), torch.cuda.graph(graph, capture_error_mode=CAPTURE):
            for _ in range(calls):
                fn()
    except RuntimeError:
        synchronize(device)
        graph.reset()
        return best_ms(fn, device, iters)
    try:
        return best_ms(graph.replay, device, iters) / calls
    finally:
        synchronize(device)
        graph.reset()
