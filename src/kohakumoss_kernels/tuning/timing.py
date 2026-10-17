"""Device-time measurement: busy warmup, L2 flush between calls, CUDA-graph replay."""

import time
from collections.abc import Callable

import torch

from kohakumoss_kernels.hardware.current import on_device

WARMUP_MS = 25.0
CAPTURE = "thread_local"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


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


def l2_flush(device: torch.device, l2_bytes: int) -> torch.Tensor:
    """A buffer larger than L2; zeroing it evicts the previous call's operands."""
    return torch.empty(
        max(2 * l2_bytes, 8 << 20) // 4, device=device, dtype=torch.int32
    )


def graph_ms(
    fn: Callable[[], object], device: torch.device, calls: int = 5, iters: int = 10
) -> float:
    """Device ms per call replayed from a CUDA graph of `calls` launches (no host
    launch cost); falls back to plain timing when capture fails or off CUDA."""
    fn()
    _sync(device)
    if device.type != "cuda":
        return best_ms(fn, device, iters)
    graph = torch.cuda.CUDAGraph()
    try:
        with on_device(device), torch.cuda.graph(graph, capture_error_mode=CAPTURE):
            for _ in range(calls):
                fn()
    except RuntimeError:
        _sync(device)
        return best_ms(fn, device, iters)
    return best_ms(graph.replay, device, iters) / calls
