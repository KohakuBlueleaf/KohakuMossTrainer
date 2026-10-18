"""Muon + AdamW hybrid in one optimizer (one param_groups list for the schedule).

Groups with `use_muon` take Nesterov momentum then a 5-step Newton-Schulz
orthogonalisation per chunk (fused qkv / up-gate matrices are split by `splits`),
scaled by 0.2 * sqrt(max(rows, cols)). Other groups run AdamW. See
docs/concepts/training.md.
"""

import math
from functools import partial

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.training.optim.optimizers.skip import (
    store,
    store_unless,
    working,
    working_copy,
)

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal factor of `g` (2-D) in bf16 iterations."""
    a, b, c = NS_COEFFS
    x = g.to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class MuonHybrid(Optimizer):
    """Muon on `use_muon` groups, AdamW on the rest; decoupled weight decay on both."""

    def __init__(
        self,
        params,
        lr: float = 5e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        rms_match: float = 0.2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            rms_match=rms_match,
            use_muon=False,
            splits=None,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        self._update(working, store)
        return None

    @torch.no_grad()
    def step_masked(self, found: Tensor) -> None:
        """`step` that leaves every parameter and state tensor unchanged where the 0-d
        `found` is nonzero; host step counts still advance (the fused-step bundle puts
        them back)."""
        self._update(working_copy, partial(store_unless, found.bool()))

    def _update(self, work, put) -> None:
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon(group, work, put)
            else:
                self._adamw(group, work, put)

    def _muon(self, group: dict, work, put) -> None:
        lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
        splits = group["splits"] or [1] * len(group["params"])
        for p, n in zip(group["params"], splits):
            if p.grad is None:
                continue
            g = p.grad.float()
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = work(state["momentum_buffer"], torch.float32).mul_(mom).add_(g)
            put(state["momentum_buffer"], buf)
            upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
            chunks = upd.chunk(n, 0)
            ortho = torch.cat([newton_schulz(c, group["ns_steps"]) for c in chunks], 0)
            rows = p.shape[0] // n
            scale = group["rms_match"] * math.sqrt(max(rows, p.shape[1]))
            q = work(p, p.dtype)
            if wd:
                q.mul_(1 - lr * wd)
            q.add_(ortho.to(p.dtype), alpha=-lr * scale)
            put(p, q)

    def _adamw(self, group: dict, work, put) -> None:
        b1, b2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] += 1
            t = state["step"]
            g = p.grad.float()
            m = work(state["exp_avg"], torch.float32).mul_(b1).add_(g, alpha=1 - b1)
            v = work(state["exp_avg_sq"], torch.float32).mul_(b2)
            v.addcmul_(g, g, value=1 - b2)
            put(state["exp_avg"], m)
            put(state["exp_avg_sq"], v)
            denom = (v / (1 - b2**t)).sqrt_().add_(eps)
            pf = work(p, torch.float32)
            if wd:
                pf.mul_(1 - lr * wd)
            pf.addcdiv_(m, denom, value=-lr / (1 - b1**t))
            put(p, pf)
