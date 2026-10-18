"""Optimizer step ops (AdamW over fp32 / 16-bit states and 8-bit blockwise states;
nearest / stochastic / Kahan parameter writeback), bound once by `resolve`;
`reference` defines what each op computes. See docs/internals/kernel-dev.md#optimizer.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from kohakumoss_kernels import registry
from kohakumoss_kernels.optim import reference
from kohakumoss_kernels.optim.reference import BLOCK, MODES
from kohakumoss_kernels.tuning.buckets import cdiv

__all__ = [
    "BLOCK",
    "MODES",
    "OptimOps",
    "SEED_STRIDE",
    "looped_multi",
    "new_8bit_state",
    "reference",
    "resolve",
]

SEED_STRIDE = 1000003


def looped_multi(single: Callable) -> Callable:
    """A multi-tensor AdamW from a per-tensor one; tensor t is seeded with
    `seed + t * SEED_STRIDE`."""

    def multi(
        params, grads, ms, rs, comps, lr, b1, b2, eps, wd, step, mode="nearest", seed=0
    ):
        comps = comps if comps is not None else [None] * len(params)
        for t, (p, g, m, r, c) in enumerate(zip(params, grads, ms, rs, comps)):
            s = (seed + t * SEED_STRIDE) & 0xFFFFFFFF
            single(p, g, m, r, c, lr, b1, b2, eps, wd, step, mode, s)

    return multi


registry.register("adamw_step", "eager", reference.adamw_step)
registry.register("adamw8bit_step", "eager", reference.adamw8bit_step)
registry.register("adamw_step_multi", "eager", looped_multi(reference.adamw_step))


@dataclass(frozen=True)
class OptimOps:
    """Bound step callables: `adamw(p, g, m, r, comp, lr, b1, b2, eps, wd, step, mode,
    seed)`, `adamw8bit(p, g, mq, ms, rq, rs, comp, ...)` and `adamw_multi(params,
    grads, ms, rs, comps, ...)` (all in place)."""

    adamw: Callable
    adamw8bit: Callable
    adamw_multi: Callable


def resolve(device, dtype: torch.dtype, backend: str | None = None) -> OptimOps:
    """Implementations for parameters of `dtype` on `device`, chosen once."""
    return OptimOps(
        registry.resolve("adamw_step", device, dtype, backend=backend),
        registry.resolve("adamw8bit_step", device, dtype, backend=backend),
        registry.resolve("adamw_step_multi", device, dtype, backend=backend),
    )


def new_8bit_state(p: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Zero (m codes int8, m scales, r codes uint8, r scales) for parameter `p`."""
    n = p.numel()
    blocks = cdiv(n, BLOCK)
    dev = p.device
    return (
        torch.zeros(n, device=dev, dtype=torch.int8),
        torch.full((blocks,), 1e-30, device=dev, dtype=torch.float32),
        torch.zeros(n, device=dev, dtype=torch.uint8),
        torch.full((blocks,), 1e-30, device=dev, dtype=torch.float32),
    )
