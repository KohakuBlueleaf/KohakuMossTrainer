"""Build the OptimBundle from knobs: groups -> optimiser -> schedule -> scaler -> bundle."""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor

from kohakumoss.precision.scaler import DynamicScaler, NoScaler
from kohakumoss.training.optim.bundle.base import Bundle, local_grad_norm
from kohakumoss.training.optim.bundle.fused import FusedStepBundle
from kohakumoss.training.optim.bundle.mirror import schedule_for
from kohakumoss.training.optim.optimizers.factory import OPTIMIZERS
from kohakumoss.types.plans import PrecisionConfig
from kohakumoss_common.calls import identity, nothing
from kohakumoss_common.registry import build

SHARED_STEP = frozenset({"master_adamw", "host_state_adamw", "cpu_adamw"})


def build_scaler(prec: PrecisionConfig):
    if not prec.scaler:
        return NoScaler()
    return DynamicScaler(prec.scaler_init, prec.scaler_growth_interval)


def _mode_fns(optimizer) -> tuple[Callable, Callable]:
    """train()/eval() switches for schedule-free optimisers; no-ops otherwise."""
    if hasattr(optimizer, "train") and hasattr(optimizer, "eval"):
        return optimizer.train, optimizer.eval
    return nothing, nothing


def build_bundle(
    groups: list[dict],
    device: torch.device,
    optimizer: str | dict,
    hyper: dict,
    schedule: dict,
    steps: int,
    prec: PrecisionConfig,
    grad_clip: float,
    fused_step: bool = False,
    grad_norm: Callable[[list[Tensor]], Tensor] = local_grad_norm,
    agree: Callable[[Tensor], Tensor] = identity,
    **hooks,
):
    """The bundle the drivers use; `fused_step` returns the per-parameter variant.
    `hooks`: `before` / `after` (around the optimiser step) and `reduce_loss`."""
    scaler = build_scaler(prec)
    make = partial(build, optimizer, OPTIMIZERS, device=device, **hyper)
    if fused_step:
        shared = make(groups=groups) if optimizer in SHARED_STEP else None
        bundle = FusedStepBundle(
            build_one=lambda one: make(groups=one),
            schedule_factory=lambda opt: schedule_for(opt, schedule, steps),
            scaler=scaler,
            groups=groups,
            agree=agree,
            clip=grad_clip,
            shared=shared,
            device=device,
            **{k: v for k, v in hooks.items() if k not in ("grads", "grad_names")},
        )
        bundle.train_mode()
        return bundle
    opt = make(groups=groups)
    train_mode, eval_mode = _mode_fns(opt)
    train_mode()
    return Bundle(
        optimizer=opt,
        schedule=schedule_for(opt, schedule, steps),
        scaler=scaler,
        groups=groups,
        grad_clip=grad_clip,
        grad_norm=grad_norm,
        agree=agree,
        train_mode=train_mode,
        eval_mode=eval_mode,
        device=device,
        **hooks,
    )
