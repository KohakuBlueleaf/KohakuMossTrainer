"""Heterogeneous balancing: measured device speed -> per-rank token budgets and
contiguous pipeline stage splits."""

import time

import torch
import torch.distributed as dist


def measure_speed(device: torch.device | str, n: int = 2048, iters: int = 10) -> float:
    """Dense matmul TFLOP/s of `device` in its fastest half dtype (fp32 on CPU)."""
    device = torch.device(device)
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
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    _sync(device)
    dt = (time.perf_counter() - t0) / iters
    return 2 * n**3 / dt / 1e12


def _sync(device: torch.device) -> None:
    match device.type:
        case "cuda":
            torch.cuda.synchronize(device)
        case "xpu":
            torch.xpu.synchronize(device)


def gather_speeds(speed: float, device: torch.device, group=None) -> list[float]:
    """Every rank's measured speed (collective)."""
    t = torch.tensor([speed], dtype=torch.float32, device=device)
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, t, group=group)
    return [float(x.item()) for x in out]


def token_budgets(total: int, speeds: list[float], multiple: int = 256) -> list[int]:
    """Split `total` tokens across ranks proportionally to speed, in `multiple`s."""
    s = sum(speeds)
    raw = [total * v / s for v in speeds]
    out = [max(multiple, int(r // multiple) * multiple) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    i = 0
    while sum(out) + multiple <= total:
        out[order[i % len(order)]] += multiple
        i += 1
    return out


def split_layers(
    costs: list[float], speeds: list[float], caps: list[float] | None = None
) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) per stage minimising the slowest stage's cost / speed.

    `caps[s]` bounds the summed cost on stage s (memory). Exact DP over cut points.
    """
    n, k = len(costs), len(speeds)
    if k > n:
        raise ValueError("more stages than layers")
    prefix = [0.0]
    for c in costs:
        prefix.append(prefix[-1] + c)
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for s in range(1, k + 1):
        for j in range(s, n - (k - s) + 1):
            for i in range(s - 1, j):
                load = prefix[j] - prefix[i]
                if caps is not None and load > caps[s - 1]:
                    continue
                val = max(best[s - 1][i], load / speeds[s - 1])
                if val < best[s][j]:
                    best[s][j], cut[s][j] = val, i
    if best[k][n] == inf:
        raise ValueError("no split satisfies the stage caps")
    bounds, j = [], n
    for s in range(k, 0, -1):
        i = cut[s][j]
        bounds.append((i, j))
        j = i
    return bounds[::-1]
