"""Pipeline stage split: per-layer costs (measured forward + backward at the pipe shape,
an analytic FLOP model, or uniform) plus the prologue / epilogue costs, partitioned
into the contiguous layer ranges the schedule's stage placement needs. See
docs/guides/multi-gpu.md (stage split)."""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from kohakumoss.logging.log import get_logger, log_table
from kohakumoss.parallel.pipeline.balance import measure_speed
from kohakumoss.parallel.pipeline.distributed import (
    V_SHAPED,
    PipeShape,
    padded_layout,
    stage_ids,
    stage_owner,
)
from kohakumoss.parallel.pipeline.parts import PartsCtx
from kohakumoss_kernels.hardware.current import rng_kept

log = get_logger(__name__)
SPLITS = ("measured", "flops", "even")


@dataclass
class StageCosts:
    """Cost of every layer, of the prologue (first stage) and the epilogue (last
    stage), in one unit (ms when measured), and every layer's parameter count."""

    layers: list[float]
    first: float
    last: float
    params: list[float]
    unit: str = "ms"


def uniform_costs(params: list[float]) -> StageCosts:
    """Every layer costs 1, no prologue / epilogue cost."""
    return StageCosts([1.0] * len(params), 0.0, 0.0, list(params), "layer")


def _param_counts(parts) -> list[float]:
    return [float(sum(p.numel() for p in b.parameters())) for b in parts.layers]


def flop_costs(parts, shape: PipeShape) -> StageCosts:
    """Forward FLOPs of each block, the prologue (text blocks on the text tokens,
    embedders) and the epilogue (head) at the probe micro-batch."""
    cfg = parts.cfg
    hw, text_len = shape.probe()
    img = int((hw[:, 0] * hw[:, 1]).sum())
    txt = int(text_len.sum())
    joint = [int(h * w + t) for (h, w), t in zip(hw.tolist(), text_len.tolist())]
    d, f = cfg.dim, cfg.ffn_dim
    linear = 2 * d * (4 * d + 3 * f)
    block = shape.tokens * linear + 4 * d * sum(n * n for n in joint)
    text_attn = 4 * d * sum(t * t for t in text_len.tolist())
    text = cfg.text_blocks * (txt * linear + text_attn)
    first = text + 2 * txt * cfg.text_dim * d + 2 * img * cfg.latent_channels * d
    last = 2 * img * d * cfg.latent_channels
    layers = [float(block)] * len(parts.layers)
    return StageCosts(layers, float(first), float(last), _param_counts(parts), "flop")


def _timer(device: torch.device) -> Callable[[Callable], float]:
    """`time(fn)` -> wall ms of one call, synchronised on `device`."""
    if device.type != "cuda":

        def host(fn) -> float:
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) * 1e3

        return host

    def cuda(fn) -> float:
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        return a.elapsed_time(b)

    return cuda


def _median(fn, timer, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    return statistics.median(timer(fn) for _ in range(iters))


def measure_costs(
    parts,
    shape: PipeShape,
    device: torch.device,
    autocast: Callable,
    runner,
    warmup: int = 2,
    iters: int = 3,
) -> StageCosts:
    """Median ms of forward + backward of every block (through `runner`, as training
    runs it), the prologue and the epilogue at the probe micro-batch on `device`; a
    block's parameter gradients are cleared once it is timed (one block's exist at a
    time), every gradient it leaves is cleared, and the caller's CPU and `device`
    random generators are left where they were."""
    with rng_kept([device]):
        return _measured(parts, shape, device, autocast, runner, warmup, iters)


def _measured(parts, shape, device, autocast, runner, warmup, iters) -> StageCosts:
    cfg = parts.cfg
    timer = _timer(device)
    hw, text_len = shape.probe()
    used = int((hw[:, 0] * hw[:, 1]).sum() + text_len.sum())
    seq = padded_layout(hw, text_len, None, used, shape).to(device)
    grad = parts.conditioning_trainable()
    dtype = parts.residual_dtype
    x = torch.randn(shape.tokens, shape.dim, device=device, dtype=dtype)
    x.requires_grad_()
    mod = 0.1 * torch.randn(shape.slots, 6, shape.dim, device=device)
    head = 0.1 * torch.randn(shape.slots, 2, shape.dim, device=device)
    mod.requires_grad_(grad)
    head.requires_grad_(grad)
    ctx = PartsCtx(seq, mod, parts.bands, parts.residual_scale, head)
    dy = torch.randn_like(x)

    def layer(i):
        def run():
            with autocast():
                y = runner.run(i, parts.layers[i], x, ctx)
            y.backward(dy.to(y.dtype))

        return run

    def timed_layer(i):
        ms = _median(layer(i), timer, warmup, iters)
        parts.layers[i].zero_grad(set_to_none=True)
        return ms

    layers = [timed_layer(i) for i in range(len(parts.layers))]
    img = torch.randn(seq.n_img, cfg.latent_channels, device=device)
    text = torch.randn(seq.n_txt, cfg.text_dim, device=device, dtype=dtype)
    sigma = torch.rand(seq.batch, device=device)

    def prologue():
        with autocast():
            x0, c = parts.prologue(img, text, sigma, seq)
        outs = [t for t in (x0, c.mod, c.head) if t.requires_grad]
        if outs:
            torch.autograd.backward(outs, [torch.ones_like(t) for t in outs])

    n = int((hw[:, 0] * hw[:, 1]).sum())
    target = torch.randn(n, cfg.latent_channels, device=device)

    def epilogue():
        with autocast():
            pred = parts.epilogue(x, ctx)[:n]
        (pred.float() - target).square().mean().backward()

    first = _median(prologue, timer, warmup, iters)
    last = _median(epilogue, timer, warmup, iters)
    parts.zero_grad(set_to_none=True)
    return StageCosts(layers, first, last, _param_counts(parts))


def broadcast_costs(costs: StageCosts, device, src: int = 0, group=None) -> StageCosts:
    """Rank `src`'s costs on every rank (collective): ranks that disagreed would build
    mismatched stages."""
    flat = torch.tensor([*costs.layers, costs.first, costs.last], device=device)
    dist.broadcast(flat, src=src, group=group)
    values = flat.tolist()
    depth = len(costs.layers)
    return StageCosts(
        values[:depth], values[depth], values[depth + 1], costs.params, costs.unit
    )


def rank_speeds(device, world: int, group=None) -> list[float]:
    """Relative speed of every rank: all 1.0 on identical devices, else each device's
    measured dense-matmul TFLOP/s (collective)."""
    if world == 1:
        return [1.0]
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    names = [None] * world
    dist.all_gather_object(names, name, group=group)
    if len(set(names)) == 1:
        return [1.0] * world
    speeds = [None] * world
    dist.all_gather_object(speeds, measure_speed(device), group=group)
    return [float(s) for s in speeds]


def _prefix(values: list[float]) -> list[float]:
    out = [0.0]
    for v in values:
        out.append(out[-1] + v)
    return out


def partition(
    costs: StageCosts, n_stages: int, speeds: list[float] | None = None
) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) per stage minimising the slowest stage's (layers + its
    prologue / epilogue) cost / speed (exact DP over cut points); among equal
    bottlenecks of a prefix the smaller sum of squared stage parameter counts wins."""
    depth = len(costs.layers)
    if n_stages > depth:
        raise ValueError(f"cannot split {depth} layers into {n_stages} stages")
    speeds = speeds or [1.0] * n_stages
    pc, pp = _prefix(costs.layers), _prefix(costs.params)
    inf = (float("inf"), float("inf"))
    best = [[inf] * (depth + 1) for _ in range(n_stages + 1)]
    cut = [[0] * (depth + 1) for _ in range(n_stages + 1)]
    best[0][0] = (0.0, 0.0)
    for s in range(1, n_stages + 1):
        extra = costs.first if s == 1 else 0.0
        extra += costs.last if s == n_stages else 0.0
        for j in range(s, depth - (n_stages - s) + 1):
            for i in range(s - 1, j):
                prev = best[s - 1][i]
                if prev == inf:
                    continue
                load = (pc[j] - pc[i] + extra) / speeds[s - 1]
                held = pp[j] - pp[i]
                key = (max(prev[0], load), prev[1] + held * held)
                if key < best[s][j]:
                    best[s][j], cut[s][j] = key, i
    bounds, j = [], depth
    for s in range(n_stages, 0, -1):
        i = cut[s][j]
        bounds.append((i, j))
        j = i
    return bounds[::-1]


def partition_v(
    costs: StageCosts, world: int, speeds: list[float] | None = None
) -> list[tuple[int, int]]:
    """[lo, hi) for the 2 * world stages of a V placement (rank r holds stages r and
    2 * world - 1 - r): each rank's layer count from the per-rank partition (rank 0
    carries prologue and epilogue), split evenly between its two stages."""
    per_rank = StageCosts(costs.layers, costs.first + costs.last, 0.0, costs.params)
    counts = [hi - lo for lo, hi in partition(per_rank, world, speeds)]
    if any(c < 2 for c in counts):
        raise ValueError(f"V placement needs >= 2 layers per rank, got {counts}")
    front = [(c + 1) // 2 for c in counts]
    back = [c - f for c, f in zip(counts, front)]
    sizes = front + back[::-1]
    return bounds_from_counts(sizes)


def bounds_from_counts(counts) -> list[tuple[int, int]]:
    """[lo, hi) per stage from layers per stage."""
    out, lo = [], 0
    for n in counts:
        out.append((lo, lo + int(n)))
        lo += int(n)
    return out


def stage_bounds(
    split,
    parts,
    shape: PipeShape,
    schedule: str,
    rank: int,
    world: int,
    device: torch.device,
    autocast: Callable,
    runner,
    per_rank: int = 1,
    group=None,
) -> list[tuple[int, int]]:
    """The layer ranges of every virtual stage under `split`: "measured" (rank 0
    times every layer, broadcast), "flops", "even", or explicit layers per stage;
    the split table is logged."""
    depth = len(parts.layers)
    n_stages = world * len(stage_ids(0, world, schedule, per_rank))
    params = _param_counts(parts)
    match split:
        case "measured":
            costs = uniform_costs(params)
            if rank == 0:
                costs = measure_costs(parts, shape, device, autocast, runner)
            if world > 1:
                costs = broadcast_costs(costs, device, 0, group)
            costs.params, costs.unit = params, "ms"
        case "flops":
            costs = flop_costs(parts, shape)
        case "even":
            costs = uniform_costs(params)
        case _:
            counts = [int(n) for n in split]
            if len(counts) != n_stages or sum(counts) != depth:
                raise ValueError(
                    f"PIPELINE_SPLIT {tuple(counts)}: need {n_stages} stage sizes "
                    f"summing to {depth} layers"
                )
            bounds = bounds_from_counts(counts)
            describe(bounds, uniform_costs(params), [1.0] * world, schedule, world)
            return bounds
    speeds = rank_speeds(device, world, group)
    if schedule in V_SHAPED:
        bounds = partition_v(costs, world, speeds)
    else:
        owners = [speeds[stage_owner(s, world, schedule)] for s in range(n_stages)]
        bounds = partition(costs, n_stages, owners)
    describe(bounds, costs, speeds, schedule, world)
    return bounds


def describe(bounds, costs: StageCosts, speeds, schedule: str, world: int) -> None:
    """One table: per virtual stage its rank, layers, cost (with prologue /
    epilogue) and parameters."""
    rows = []
    n = len(bounds)
    pc = _prefix(costs.layers)
    for s, (lo, hi) in enumerate(bounds):
        r = stage_owner(s, world, schedule)
        cost = pc[hi] - pc[lo] + (costs.first if s == 0 else 0.0)
        cost += costs.last if s == n - 1 else 0.0
        params = round(sum(costs.params[lo:hi]) / 1e6, 1)
        rows.append((s, r, f"{lo}..{hi - 1}", hi - lo, round(cost, 3), params))
    title = (
        f"pipeline split ({schedule}, {costs.unit}; prologue {costs.first:.3g}, "
        f"epilogue {costs.last:.3g}; rank speeds {[round(v, 2) for v in speeds]})"
    )
    log_table(log, title, ("stage", "rank", "layers", "n", "cost", "M params"), rows)
