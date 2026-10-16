"""Fused per-parameter optimiser steps, driven from backward hooks by the memory engine.

`step_param(p)` unscales a parameter's grad, zeroes it when non-finite (the flag is
accumulated on device), clips it by its own norm, steps, and frees the grad. The update
is one shared optimiser's `step_param` (offload AdamWs) or a one-parameter torch
optimiser per parameter, chosen at construction. The hook object (`attach`: the
engine's FusedStep, or a GradOffload that finishes host-accumulated grads) is armed on
the last micro-batch of a step only, so gradient accumulation still works.
"""

import contextlib
from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.training.optim.bundle import _noop, no_agree


def _no_clip(g: Tensor) -> None:
    return None


def _clip_param(max_norm: float, g: Tensor) -> None:
    g.mul_((max_norm / (g.float().norm() + 1e-6)).clamp(max=1.0).to(g.dtype))


class _LRHolder(Optimizer):
    """Carries one lr per group for the schedule; never steps."""

    def __init__(self, groups: list[dict]):
        holder = [{"params": [torch.zeros(1)], "lr": g["lr"]} for g in groups]
        super().__init__(holder, {"lr": groups[0]["lr"] if groups else 0.0})

    def step(self, closure=None):
        return None


class FusedStepBundle:
    """OptimBundle whose optimiser step happens per parameter during backward.

    `shared`: an optimiser over every parameter exposing `step_param(p)`; otherwise
    `build_one(groups)` builds one optimiser per parameter.
    """

    def __init__(
        self,
        build_one: Callable[[list[dict]], Optimizer],
        schedule_factory: Callable[[Optimizer], object],
        scaler,
        groups: list[dict],
        agree: Callable[[Tensor], Tensor] = no_agree,
        before: Callable[[], None] = _noop,
        after: Callable[[], None] = _noop,
        reduce_loss: Callable[[Tensor], Tensor] = no_agree,
        clip: float = 0.0,
        shared=None,
    ):
        self.scaler = scaler
        self.agree = agree
        self.clip_param = partial(_clip_param, clip) if clip > 0 else _no_clip
        self.hooks = None
        self._arm = _no_arm
        self.before = before
        self.after = after
        self.reduce_loss = reduce_loss
        self.params = [p for g in groups for p in g["params"]]
        self.shared = shared
        if shared is not None:
            self.lr_source = shared
            self.schedule = schedule_factory(shared)
            self.update = shared.step_param
            self.opts = {}
        else:
            self.lr_source = _LRHolder(groups)
            self.schedule = schedule_factory(self.lr_source)
            self.opts: dict[int, tuple[Optimizer, int]] = {}
            for gi, g in enumerate(groups):
                for p in g["params"]:
                    one = {k: v for k, v in g.items() if k != "params"}
                    one["params"] = [p]
                    self.opts[id(p)] = (build_one([one]), gi)
            self.update = self._own_update
        self.device = self.params[0].device
        self._reset()
        self.last_loss = float("nan")
        self.last_norm = float("nan")
        self.last_ok = True
        self.skipped = 0

    def attach(self, hooks, finishes: bool = False) -> None:
        """The hook object this bundle arms. `finishes`: a GradOffload whose final
        callback on the last micro-batch is `step_param`; else a FusedStep."""
        self.hooks = hooks
        self._arm = (
            partial(_arm_offload, hooks, self.step_param) if finishes else hooks.arm
        )

    def arm(self, last: bool):
        """Accumulate context: step parameters in this backward only on the last one."""
        self._arm(last)
        return contextlib.nullcontext()

    def _reset(self) -> None:
        self._bad = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._loss = torch.zeros((), device=self.device)

    def backward(self, loss: Tensor) -> None:
        self.scaler.scale_loss(loss).backward()
        self._loss = self._loss + loss.detach().float()

    def _own_update(self, param: torch.nn.Parameter) -> None:
        opt, gi = self.opts[id(param)]
        opt.param_groups[0]["lr"] = self.lr_source.param_groups[gi]["lr"]
        opt.step()

    @torch.no_grad()
    def step_param(self, param: torch.nn.Parameter) -> None:
        """Called by the engine's post-accumulate-grad hook (grad is present)."""
        g = param.grad
        self.scaler.unscale_([g])
        finite = torch.isfinite(g).all()
        g.mul_(finite.to(g.dtype)).nan_to_num_(0.0, 0.0, 0.0)
        self._bad += (~finite).to(torch.int32).to(self._bad.device)
        self.clip_param(g)
        self.update(param)
        param.grad = None

    def step(self) -> bool:
        self.before()
        loss = self.reduce_loss(self._loss.reshape(1).clone())
        bad = self.agree(
            (self._bad > 0).to(torch.int32) | (~torch.isfinite(loss)).to(torch.int32)
        )
        loss_v, bad_v = torch.cat([loss.reshape(1).float(), bad.float()]).tolist()
        ok = bad_v == 0
        self.skipped += int(not ok)
        self.after()
        self.schedule.step()
        self.scaler.update(not ok)
        self._reset()
        self.last_loss, self.last_ok = loss_v, ok
        return ok

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @property
    def lr(self) -> float:
        return float(self.lr_source.param_groups[0]["lr"])

    def train_mode(self) -> None:
        return None

    def eval_mode(self) -> None:
        return None

    def _optimizers(self) -> list:
        return (
            [self.shared]
            if self.shared is not None
            else [o for o, _ in self.opts.values()]
        )

    def state_dict(self) -> dict:
        return {
            "optimizers": [opt.state_dict() for opt in self._optimizers()],
            "schedule": self.schedule.state_dict(),
            "scaler": self.scaler.state_dict(),
            "skipped": self.skipped,
        }

    def load_state_dict(self, state: dict) -> None:
        for opt, sd in zip(self._optimizers(), state["optimizers"]):
            opt.load_state_dict(sd)
        self.schedule.load_state_dict(state["schedule"])
        self.scaler.load_state_dict(state["scaler"])
        self.skipped = state.get("skipped", 0)


def _no_arm(last: bool) -> None:
    raise RuntimeError("FusedStepBundle has no hook object attached")


def _arm_offload(hooks, final, last: bool) -> None:
    hooks.arm(last, final)
