"""Heterogeneous balancing: the measured dense-matmul speed of a device (the stage
split's rank speeds, `split.rank_speeds`)."""

import time

import torch

from kohakumoss_kernels.hardware.current import rng_kept, synchronize


def measure_speed(device: torch.device | str, n: int = 2048, iters: int = 10) -> float:
    """Dense matmul TFLOP/s of `device` in its fastest half dtype (fp32 on CPU); the
    caller's CPU and `device` random generators are left where they were."""
    device = torch.device(device)
    with rng_kept([device]):
        return _matmul_tflops(device, n, iters)


def _matmul_tflops(device: torch.device, n: int, iters: int) -> float:
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    elif device.type == "xpu":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        n = min(n, 512)
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(2):
        a @ b
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    return 2 * n**3 / dt / 1e12
