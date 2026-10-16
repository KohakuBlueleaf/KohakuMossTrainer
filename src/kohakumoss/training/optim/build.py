"""Build the OptimBundle from knobs: groups -> optimiser -> schedule -> scaler -> bundle."""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor

from kohakumoss.registry import build
from kohakumoss.training.optim.bundle import Bundle, local_grad_norm, no_agree
from kohakumoss.training.optim.factory import OPTIMIZERS
from kohakumoss.training.optim.fused import FusedStepBundle
from kohakumoss.training.optim.mirror import schedule_for
from kohakumoss.training.precision.scaler import DynamicScaler, NoScaler
from kohakumoss.types.plans import PrecisionPolicy

SHARED_STEP = frozenset({"master_adamw", "host_state_adamw", "cpu_adamw"})


def build_scaler(policy: PrecisionPolicy):
    if not policy.scaler:
        return NoScaler()
    return DynamicScaler(policy.scaler_init, policy.scaler_growth_interval)


def _mode_fns(optimizer) -> tuple[Callable, Callable]:
    """train()/eval() switches for schedule-free optimisers; no-ops otherwise."""
    if hasattr(optimizer, "train") and hasattr(optimizer, "eval"):
        return optimizer.train, optimizer.eval
    return (lambda: None), (lambda: None)


def build_bundle(
    groups: list[dict],
    device: torch.device,
    optimizer: str | dict,
    hyper: dict,
    schedule: dict,
    steps: int,
    policy: PrecisionPolicy,
    grad_clip: float,
    fused_step: bool = False,
    grad_norm: Callable[[list[Tensor]], Tensor] = local_grad_norm,
    agree: Callable[[Tensor], Tensor] = no_agree,
    **hooks,
):
    """The bundle the drivers use; `fused_step` returns the per-parameter variant.
    `hooks`: `before` / `after` (around the optimiser step) and `reduce_loss`."""
    scaler = build_scaler(policy)
    make = partial(build, optimizer, OPTIMIZERS, device=device, **hyper)
    if fused_step:
        shared = make(groups=groups) if optimizer in SHARED_STEP else None
        return FusedStepBundle(
            build_one=lambda one: make(groups=one),
            schedule_factory=lambda opt: schedule_for(opt, schedule, steps),
            scaler=scaler,
            groups=groups,
            agree=agree,
            clip=grad_clip,
            shared=shared,
            **{k: v for k, v in hooks.items() if k != "grads"},
        )
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
        **hooks,
    )
