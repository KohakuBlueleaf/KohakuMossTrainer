"""Optimiser wiring of a MemoryPlan: where each trainable parameter's AdamW runs, what
steps inside backward, and the OptimBundle over it. The trainer and MemorySession both
build here. See docs/internals/offload.md (optimisers)."""

import contextlib
from collections.abc import Callable

import torch

from kohakumoss.offload.stream_optim import StreamedAdamW
from kohakumoss.training.memory.apply import (
    MemoryEngines,
    attach_fused_step,
    attach_grad_offload,
)
from kohakumoss.training.optim.build import build_bundle
from kohakumoss.training.optim.bundle import local_grad_norm, no_agree
from kohakumoss.types.plans import MemoryPlan, PrecisionPolicy

ADAMW_BY_STATE = {
    "gpu": "adamw",
    "8bit": "adamw8bit",
    "cpu": "cpu_adamw",
    "stream": "host_state_adamw",
}
HOST_BUFFER_OPTIMIZERS = ("cpu_adamw", "host_state_adamw")
GROUP_LIST_KEYS = ("params", "names", "splits")


def optimizer_for(optimizer: str, plan: MemoryPlan) -> str:
    """OPTIMIZER, with `adamw` following `plan.optim_state` (knobs reject a non-GPU
    optim_state for every other optimiser)."""
    return ADAMW_BY_STATE[plan.optim_state] if optimizer == "adamw" else optimizer


def split_groups(groups: list[dict], streamed: set[int]) -> tuple[list, list]:
    """(device groups, streamed-block groups): each group's parameters split by id,
    every other key kept; empty parts dropped."""
    dev, blocks = [], []
    for g in groups:
        for out, keep in ((dev, False), (blocks, True)):
            idx = [i for i, p in enumerate(g["params"]) if (id(p) in streamed) == keep]
            if not idx:
                continue
            part = dict(g)
            for key in GROUP_LIST_KEYS:
                if key in g:
                    part[key] = [g[key][i] for i in idx]
            out.append(part)
    return dev, blocks


def _chain(*fns) -> Callable[[], None]:
    fns = tuple(f for f in fns if f is not None)

    def run() -> None:
        for f in fns:
            f()

    return run


def _offload_parts(grads, host_optimizer: bool, fused: bool) -> dict:
    """Bundle hooks for gradient offload: in-backward finish, host optimiser (reads the
    host sums), or device optimiser (sums put back on the device before the step)."""
    host = list(grads.host.values())
    match (fused, host_optimizer):
        case (True, _):
            return {"after": grads.zero}
        case (False, True):
            return {"before": grads.flush, "after": grads.zero, "grads": lambda: host}
        case _:
            return {"before": _chain(grads.flush, grads.to_device), "after": grads.zero}


def _offload_arm(grads, accumulate: Callable):
    """Offload every micro-batch's gradients and keep the gradient-sync context."""
    if grads is None:
        return accumulate

    def arm(last: bool):
        grads.arm(last, None)
        return accumulate(last)

    return arm


def _host_optimizer(bundle):
    return (
        bundle.shared
        if getattr(bundle, "shared", None) is not None
        else bundle.optimizer
    )


def _block_groups(groups: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in g.items() if k not in ("names", "splits", "use_muon")}
        for g in groups
    ]


def build_memory_optim(
    groups: list[dict],
    plan: MemoryPlan,
    engines: MemoryEngines,
    device: torch.device,
    optimizer: str,
    hyper: dict,
    schedule: dict,
    steps: int,
    policy: PrecisionPolicy,
    grad_clip: float,
    hooks=None,
):
    """(bundle, accumulate-context factory) over `groups` under `plan`.

    `swap` blocks with a GPU optim_state step per block in backward (StreamedAdamW;
    resident states for `gpu`); with `cpu` they join the host AdamW after
    `flush_grads`. `hooks`: the parallel layout's loop hooks (ParallelHooks)."""
    name = optimizer_for(optimizer, plan)
    stream = engines.stream
    streamed = {id(p) for p in engines.host_params()}
    gpu_blocks = bool(streamed) and plan.optim_state != "cpu"
    fused = bool(plan.fused_step)
    if gpu_blocks and not fused:
        raise ValueError(
            "swap blocks with a GPU optimiser step in backward: fused_step"
        )
    dev_groups, block_groups = (
        split_groups(groups, streamed) if gpu_blocks else (groups, [])
    )
    if gpu_blocks and not dev_groups:
        raise ValueError(
            "swap with a GPU optimiser needs a trainable non-block parameter"
        )
    device_params = [
        p for g in dev_groups for p in g["params"] if id(p) not in streamed
    ]
    grads = None
    extra = {}
    if plan.grad_offload:
        grads = attach_grad_offload(engines, device_params, plan.pinned)
        extra = _offload_parts(grads, name == "cpu_adamw", fused)
    if name in HOST_BUFFER_OPTIMIZERS:
        hyper = {**hyper, "pinned": plan.pinned}
    before = getattr(hooks, "before", None)
    bundle = build_bundle(
        dev_groups,
        device,
        name,
        hyper,
        schedule,
        steps,
        policy,
        grad_clip,
        fused_step=fused,
        grad_norm=getattr(hooks, "grad_norm", local_grad_norm),
        agree=getattr(hooks, "agree", no_agree),
        before=_chain(before, engines.before_step, extra.get("before")),
        after=_chain(engines.after_step, extra.get("after")),
        reduce_loss=getattr(hooks, "reduce_loss", no_agree),
        grads=extra.get("grads"),
    )
    accumulate = getattr(hooks, "accumulate", None)
    if not fused:
        if name == "cpu_adamw" and grads is not None:
            _host_optimizer(bundle).use_host_grads(grads)
        return bundle, _offload_arm(grads, accumulate or _null_accumulate)
    if grads is not None:
        bundle.attach(grads, finishes=True)
    else:
        bundle.attach(attach_fused_step(engines, device_params, bundle.step_param))
    if gpu_blocks:
        blocks = StreamedAdamW(
            stream,
            lr=hyper["lr"],
            betas=hyper["betas"],
            eps=hyper["eps"],
            weight_decay=0.0,
            resident=plan.optim_state == "gpu",
            groups=_block_groups(block_groups),
        )
        bundle.attach_blocks(blocks)
    return bundle, bundle.arm


def _null_accumulate(last: bool):
    return contextlib.nullcontext()
