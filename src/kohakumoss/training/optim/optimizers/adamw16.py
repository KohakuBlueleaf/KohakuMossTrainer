"""AdamW for 16-bit parameters: fp32 math in registers, 16-bit state, sqrt(v) stored in
place of v, and a selectable parameter write-back rounding (nearest / stochastic /
Kahan). See docs/concepts/training.md.
"""

import math
from functools import partial

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.training.optim.optimizers.rounding import WRITERS
from kohakumoss.training.optim.optimizers.skip import (
    store,
    store_unless,
    working,
    working_copy,
    write_unless,
    write_unless_comp,
)


class AdamW16(Optimizer):
    """AdamW whose params and states may be fp16/bf16; math is fp32."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rounding: str = "nearest",
    ):
        if rounding not in WRITERS:
            raise ValueError(f"rounding must be one of {sorted(WRITERS)}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rounding = rounding
        self._write = WRITERS[rounding]
        self._write_unless = write_unless_comp if rounding == "kahan" else write_unless

    def _init_state(self, p: torch.Tensor) -> dict:
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sqrt"] = torch.zeros_like(p)
            if self.rounding == "kahan":
                state["comp"] = torch.zeros_like(p)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        self._update(working, store, self._write)
        return None

    @torch.no_grad()
    def step_masked(self, found: Tensor) -> None:
        """`step` that leaves every parameter, moment and compensation unchanged where
        the 0-d `found` is nonzero; the host step count still advances (the fused-step
        bundle puts it back)."""
        skip = found.bool()
        self._update(
            working_copy,
            partial(store_unless, skip),
            partial(self._write_unless, self._write, skip),
        )

    def _update(self, work, put, write) -> None:
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_state(p)
                state["step"] += 1
                t = state["step"]
                g = p.grad.float()
                m = work(state["exp_avg"], torch.float32).mul_(b1)
                m.add_(g, alpha=1 - b1)
                v = work(state["exp_avg_sqrt"], torch.float32).square_().mul_(b2)
                v.addcmul_(g, g, value=1 - b2)
                s = v.sqrt_()
                put(state["exp_avg"], m)
                put(state["exp_avg_sqrt"], s)
                bc1 = 1 - b1**t
                bc2 = math.sqrt(1 - b2**t)
                pf = work(p, torch.float32)
                if wd:
                    pf.mul_(1 - lr * wd)
                pf.addcdiv_(m, s / bc2 + eps, value=-lr / bc1)
                write(p, pf, state.get("comp"))
