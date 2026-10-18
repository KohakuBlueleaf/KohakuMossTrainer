"""Prewarm of the block ops on scratch MOSS-shaped inputs (test helper): each op forward
(+ backward) so kernel configs and kernel-vs-compile choices are measured and stored."""

import time
from collections.abc import Callable
from functools import partial

import torch

from kohakumoss_common.log import get_logger, hl, log_table
from kohakumoss_kernels.backends.compile.choice import retune_pending
from kohakumoss_kernels.bench.blockops import segment_ids
from kohakumoss_kernels.hardware.current import rng_kept, synchronize
from kohakumoss_kernels.registry import Bound, exercise

log = get_logger("kohakumoss_kernels.backends.compile.prewarm")
BLOCK_ORDER = ("rms_norm_mod", "rms_norm", "gated_residual", "situ_glu", "qk_norm_rope")


def _positions(n: int, device) -> torch.Tensor:
    pos = torch.zeros(n, 4, device=device)
    pos[:, 1] = torch.arange(n, device=device) % 512
    return pos


def prewarm_block_ops(
    ops: dict[str, Callable],
    device: torch.device | str,
    dtype: torch.dtype,
    lengths: list[list[int]],
    dim: int = 3072,
    heads: int = 24,
    ffn_dim: int = 8192,
    train_params: bool = False,
    backward: bool = True,
) -> int:
    """`ops` maps op names (registry names) to resolved callables; `lengths` lists
    batches of per-sample joint lengths; `backward=False` runs forward only under
    `no_grad`. Returns how many problems were measured."""
    device = torch.device(device)
    hd = dim // heads
    bands = 1 / (10000.0 ** (torch.arange(hd // 8, device=device) / (hd // 8)))
    names = [n for n in BLOCK_ORDER if n in ops]
    total, done = len(names) * len(lengths), 0
    spent: dict[str, float] = {}
    start = time.perf_counter()
    log.info(f"prewarm block ops on {device}: {len(names)} ops x {len(lengths)} sizes")
    grad, run = (
        (torch.enable_grad, _forward_backward)
        if backward
        else (torch.no_grad, _forward)
    )
    train_params = train_params and backward
    with grad(), rng_kept([device]):
        for lens in lengths:
            n, b = sum(lens), len(lens)
            seg = segment_ids(lens, device)
            x = torch.randn(n, dim, device=device).to(dtype).requires_grad_(backward)
            mod = torch.randn(b, dim, device=device).requires_grad_(train_params)
            w = torch.ones(dim, device=device).requires_grad_(train_params)
            calls = {
                "rms_norm_mod": (x, w, mod, mod, seg, 1e-6),
                "rms_norm": (x, w, 1e-6),
                "gated_residual": (x, x, mod, seg, 1.0),
                "situ_glu": (
                    torch.randn(n, 2 * ffn_dim, device=device)
                    .to(dtype)
                    .requires_grad_(backward),
                ),
                "qk_norm_rope": (
                    x.view(n, heads, hd),
                    x.view(n, heads, hd),
                    torch.ones(hd, device=device),
                    torch.ones(hd, device=device),
                    _positions(n, device),
                    bands,
                    1e-6,
                ),
            }
            for name, args in calls.items():
                if name not in ops:
                    continue
                done += 1
                log.info(f"tuning {name} {hl(f'{done}/{total}')} ({hl(f'{n} tokens')})")
                step = partial(run, args=args)
                op = ops[name]
                began = time.perf_counter()
                if isinstance(op, Bound):
                    exercise(op, step)
                else:
                    step(op)
                synchronize(device)
                spent[name] = spent.get(name, 0.0) + time.perf_counter() - began
    measured = retune_pending()
    took = time.perf_counter() - start
    rows = [(name, round(s, 2)) for name, s in spent.items()]
    log_table(log, "prewarm block ops per op", ("op", "secs"), rows)
    log.info(
        f"prewarm block ops: done in {hl(f'{took:.2f}s')}, "
        f"{hl(measured)} kernel-vs-compile problems measured"
    )
    return measured


def _forward_backward(op: Callable, args: tuple) -> None:
    out = op(*args)
    outs = out if isinstance(out, tuple) else (out,)
    torch.autograd.backward([o.float().sum() for o in outs])


def _forward(op: Callable, args: tuple) -> None:
    op(*args)
