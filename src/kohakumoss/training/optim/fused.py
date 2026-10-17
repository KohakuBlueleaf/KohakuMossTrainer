"""Fused per-parameter optimiser steps, driven from backward hooks by the memory engine.
See docs/internals/offload.md (fused step, non-finite gradients)."""

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


def _no_post(p: Tensor, lr: float, found: Tensor) -> None:
    return None


def _decay_to_one(coef: float, p: Tensor, lr: float, found: Tensor) -> None:
    p.sub_((p - 1.0) * ((1.0 - found) * (lr * coef)).to(p.dtype))


def _step_plain(opt: Optimizer, found: Tensor) -> None:
    opt.step()


def _step_found_inf(opt: Optimizer, found: Tensor) -> None:
    opt.found_inf = found
    opt.step()


class _LRHolder(Optimizer):
    """Carries one lr per group for the schedule; never steps."""

    def __init__(self, groups: list[dict]):
        holder = [{"params": [torch.zeros(1)], "lr": g["lr"]} for g in groups]
        super().__init__(holder, {"lr": groups[0]["lr"] if groups else 0.0})

    def step(self, closure=None):
        return None


class FusedStepBundle:
    """OptimBundle whose optimiser step happens per parameter during backward.

    `shared`: an optimiser over every device parameter exposing `step_param(p,
    found_inf)`; otherwise `build_one(groups)` builds one optimiser per parameter.
    `attach_blocks(opt)` adds a per-block optimiser (StreamedAdamW) that steps in
    backward through the same gradient transform.
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
        device: torch.device | None = None,
    ):
        self.scaler = scaler
        self.agree = agree
        self.clip_param = partial(_clip_param, clip) if clip > 0 else _no_clip
        self.hooks = None
        self.blocks = None
        self._arm = _no_arm
        self.before = before
        self.after = after
        self.reduce_loss = reduce_loss
        self.params = [p for g in groups for p in g["params"]]
        self.shared = shared
        self.finishers: list[Callable[[], None]] = []
        self._lr0 = float(groups[0]["lr"]) if groups else 1.0
        self._block_lr: list[tuple[dict, float]] = []
        if shared is not None:
            self.lr_source = shared
            self.schedule = schedule_factory(shared)
            self.update = shared.step_param
            self.opts = {}
            if hasattr(shared, "finish"):
                self.finishers.append(shared.finish)
        else:
            self.lr_source = _LRHolder(groups)
            self.schedule = schedule_factory(self.lr_source)
            self.opts: dict[int, tuple] = {}
            for gi, g in enumerate(groups):
                post = (
                    partial(_decay_to_one, g["decay_to_one_coef"])
                    if g.get("decay_to_one")
                    else _no_post
                )
                for p in g["params"]:
                    one = {k: v for k, v in g.items() if k != "params"}
                    one["params"] = [p]
                    opt = build_one([one])
                    run = _step_found_inf if opt.defaults.get("fused") else _step_plain
                    self.opts[id(p)] = (opt, gi, run, post)
            self.update = self._own_update
        self.device = device if device is not None else self.params[0].device
        self._reset()
        self.last_loss = float("nan")
        self.last_norm = float("nan")
        self.last_ok = True
        self.skipped = 0
        self.nonfinite = 0

    def attach(self, hooks, finishes: bool = False) -> None:
        """The hook object this bundle arms. `finishes`: a GradOffload whose final
        callback on the last micro-batch is `step_param`; else a FusedStep."""
        self.hooks = hooks
        self._arm = (
            partial(_arm_offload, hooks, self.step_param) if finishes else hooks.arm
        )
        if hasattr(hooks, "finish"):
            self.finishers.insert(0, hooks.finish)

    def attach_blocks(self, blocks) -> None:
        """A per-block in-backward optimiser (StreamedAdamW) sharing the transform; its
        groups follow the schedule (lr scaled like group 0)."""
        self.blocks = blocks
        blocks.transform = self.transform
        self.finishers.append(blocks.finish)
        self._arm = partial(_arm_blocks, blocks, self._arm)
        self._block_lr = [(g, float(g["lr"])) for g in blocks.param_groups]
        self._mirror_blocks()

    def _mirror_blocks(self) -> None:
        factor = self.lr / self._lr0
        for group, lr in self._block_lr:
            group["lr"] = lr * factor

    def arm(self, last: bool):
        """Accumulate context: step parameters in this backward only on the last one."""
        self._arm(last)
        return contextlib.nullcontext()

    def _reset(self) -> None:
        self._bad = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._loss = torch.zeros((), device=self.device)
        self._sq = torch.zeros((), dtype=torch.float32, device=self.device)

    def backward(self, loss: Tensor) -> None:
        self.scaler.scale_loss(loss).backward()
        self._loss = self._loss + loss.detach().float()

    def observe(self, loss: Tensor) -> None:
        self._loss = self._loss + loss.detach().float().to(self.device)

    def _own_update(self, param: torch.nn.Parameter, found: Tensor) -> None:
        opt, gi, run, post = self.opts[id(param)]
        lr = self.lr_source.param_groups[gi]["lr"]
        opt.param_groups[0]["lr"] = lr
        run(opt, found)
        post(param, lr, found)

    @torch.no_grad()
    def transform(self, g: Tensor) -> Tensor:
        """Unscale, flag (and zero) a non-finite gradient, add its squared norm to the
        step's gradient norm, clip by its own norm; the 0-d fp32 found_inf the update
        must honour."""
        self.scaler.unscale_([g])
        finite = torch.isfinite(g).all()
        g.mul_(finite.to(g.dtype)).nan_to_num_(0.0, 0.0, 0.0)
        bad = (~finite).to(torch.int32)
        self._bad += bad.to(self._bad.device)
        self._sq += g.float().square().sum().to(self._sq.device)
        self.clip_param(g)
        return bad.float()

    @torch.no_grad()
    def step_param(self, param: torch.nn.Parameter) -> None:
        """Called by the engine's post-accumulate-grad hook (grad is present)."""
        found = self.transform(param.grad)
        self.update(param, found)
        param.grad = None

    def step(self) -> bool:
        """Finish the step. Parameters were updated inside backward (one with a
        non-finite gradient is not), so the step counts as applied: returns True;
        `nonfinite` counts steps with a non-finite gradient or loss, and `last_ok` is
        False on those. `last_norm` is the unclipped gradient norm of the step's
        finite gradients (zeroed non-finite ones add nothing)."""
        for finish in self.finishers:
            finish()
        self.before()
        loss = self.reduce_loss(self._loss.reshape(1).clone())
        bad = self.agree(
            (self._bad > 0).to(torch.int32) | (~torch.isfinite(loss)).to(torch.int32)
        )
        norm = self._sq.sqrt().reshape(1)
        loss_v, norm_v, bad_v = torch.cat(
            [loss.reshape(1).float(), norm, bad.float()]
        ).tolist()
        ok = bad_v == 0
        self.nonfinite += int(not ok)
        self.after()
        self.schedule.step()
        if self._block_lr:
            self._mirror_blocks()
        self.scaler.update(not ok)
        self._reset()
        self.last_loss, self.last_norm, self.last_ok = loss_v, norm_v, ok
        return True

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
        own = (
            [self.shared]
            if self.shared is not None
            else [o for o, *_ in self.opts.values()]
        )
        return own + ([self.blocks] if self.blocks is not None else [])

    def state_dict(self) -> dict:
        return {
            "optimizers": [opt.state_dict() for opt in self._optimizers()],
            "schedule": self.schedule.state_dict(),
            "scaler": self.scaler.state_dict(),
            "skipped": self.skipped,
            "nonfinite": self.nonfinite,
        }

    def load_state_dict(self, state: dict) -> None:
        for opt, sd in zip(self._optimizers(), state["optimizers"], strict=True):
            opt.load_state_dict(sd)
        self.schedule.load_state_dict(state["schedule"])
        if self._block_lr:
            self._mirror_blocks()
        self.scaler.load_state_dict(state["scaler"])
        self.skipped = state.get("skipped", 0)
        self.nonfinite = state.get("nonfinite", 0)


def _no_arm(last: bool) -> None:
    raise RuntimeError("FusedStepBundle has no hook object attached")


def _arm_offload(hooks, final, last: bool) -> None:
    hooks.arm(last, final)


def _arm_blocks(blocks, inner, last: bool) -> None:
    blocks.arm(last)
    inner(last)
