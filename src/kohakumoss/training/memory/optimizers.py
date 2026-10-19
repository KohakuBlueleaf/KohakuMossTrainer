"""Optimiser wiring of a MemoryPlan: where each trainable parameter's AdamW runs, what
steps inside backward, and the OptimBundle over it. The trainer and MemorySession both
build here. See docs/internals/offload.md (optimisers)."""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor

from kohakumoss.memory.apply import (
    MemoryEngines,
    attach_fused_step,
    attach_grad_offload,
)
from kohakumoss.offload.optim.stream_optim import StreamedAdamW
from kohakumoss.parallel.ddp import ShardedNorm, sum_block_stats
from kohakumoss.training.optim.bundle.base import local_grad_norm
from kohakumoss.training.optim.bundle.build import build_bundle
from kohakumoss.training.optim.bundle.nonfinite import param_names
from kohakumoss.training.optim.optimizers.factory import master_groups
from kohakumoss.types.plans import MemoryPlan, PrecisionConfig
from kohakumoss_common.calls import identity
from kohakumoss_common.weak import weak_method

ADAMW_BY_STATE = {
    "gpu": "adamw",
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


def _group_params(groups: list[dict]) -> list[Tensor]:
    return [p for g in groups for p in g["params"]]


def _host_grads(grads, masters: list[Tensor]) -> list[Tensor]:
    return grads.filled_sums() + [p.grad for p in masters if p.grad is not None]


def host_grad_names(grads, masters: list[Tensor], by_id: dict[int, str]) -> list[str]:
    """The parameter name of every `_host_grads` entry, in its order."""
    params = grads.filled_params() + [p for p in masters if p.grad is not None]
    return [by_id.get(id(p), f"host_grad[{i}]") for i, p in enumerate(params)]


def _offload_parts(
    grads, cpu_adamw: bool, fused: bool, masters, by_id: dict[int, str]
) -> dict:
    """Bundle hooks for gradient offload: in-backward finish, host optimiser (reads the
    host sums holding this step's gradients and the streamed blocks' host gradients),
    or device optimiser (sums put back on the device before the step). Off the fused
    path under DDP (a `final` on the last micro-batch), parameters whose
    last-micro-batch hook did not fire are finished first; without DDP that is a
    no-op."""
    match (fused, cpu_adamw):
        case (True, _):
            return {"after": grads.zero}
        case (False, True):
            return {
                "finish_unhooked": grads.finish_unhooked,
                "before": grads.drain,
                "after": grads.zero,
                "grads": partial(_host_grads, grads, list(masters)),
                "grad_names": partial(host_grad_names, grads, list(masters), by_id),
            }
        case _:
            return {
                "finish_unhooked": grads.finish_unhooked,
                "before": _chain(grads.drain, grads.to_device),
                "after": grads.zero,
            }


def _offload_start(grads, accumulate: Callable, final: Callable | None = None):
    """Offload every micro-batch's gradients (the last one through `final` when given)
    and keep the gradient-sync context."""
    if grads is None:
        return accumulate

    def start(last: bool):
        grads.start(last, final)
        return accumulate(last)

    return start


class _OffloadThrough:
    """A GradOffload whose in-backward finish calls `step` (the DDP reduction in front of
    the bundle's per-parameter step) instead of the bundle's own callback."""

    def __init__(self, grads, step: Callable):
        self.grads = grads
        self.step = step

    def start(self, last: bool, final: Callable | None = None) -> None:
        self.grads.start(last, self.step if final is not None else None)

    def finish(self) -> None:
        self.grads.finish()


def _ddp_kept(ddp, engines: MemoryEngines, grads, params) -> Callable | None:
    """DDP wiring of a plan that keeps gradients until the step: host-stepped streamed
    blocks are summed before their gradients leave for the host; offloaded gradients
    are summed on the last micro-batch and replace their host sums (the returned
    GradOffload callback; None without offload)."""
    if engines.host_stepped:
        owners = engines.weight_stream.owners
        reduce = ddp.block_reduce(owners, everyone=owners is None)
        ddp.watch(engines.weight_stream.reduce_with(reduce))
    if grads is None:
        return None
    ddp.host_reads = True
    return ddp.reduced(params, grads.store)


def _start_both(first: Callable, second: Callable, last: bool):
    first(last)
    return second(last)


def _not_mine(weight_stream) -> set[int]:
    """Ids of the streamed masters another rank steps (DDP block owners)."""
    return {
        id(p)
        for key, p in weight_stream.params.items()
        if not weight_stream.mine(key[0])
    }


def _owned_grads(weight_stream) -> set[int]:
    """Ids of this rank's streamed-block host gradients (sharded across ranks)."""
    return {id(g) for g in weight_stream.host_grad.values()}


def _host_barrier(sync: Callable[[], None], arena) -> None:
    """The owners' block write-backs have landed in the shared host region for every
    rank."""
    sync()
    arena.ready()


def device_optimizer(bundle):
    """The bundle's optimiser of the device (non-streamed) parameters: `shared`, else
    `optimizer`."""
    return bundle.shared if bundle.shared is not None else bundle.optimizer


def build_memory_optim(
    groups: list[dict],
    plan: MemoryPlan,
    engines: MemoryEngines,
    device: torch.device,
    optimizer: str,
    hyper: dict,
    schedule: dict,
    steps: int,
    prec: PrecisionConfig,
    grad_clip: float,
    hooks=None,
):
    """(bundle, accumulate-context factory) over `groups` under `plan`.

    `swap` blocks with a GPU optim_state step per block in backward (StreamedAdamW;
    device states for `gpu`); with `cpu` they join the host AdamW after
    `expose_grads`. `hooks`: the parallel layout's loop hooks (ParallelHooks); its
    `ddp` AsyncReducer sums every gradient that leaves backward before the step (the
    in-backward step, gradient offload, streamed blocks) across ranks first."""
    w = resolve_wiring(plan, engines, optimizer, hooks)
    by_id = {id(p): n for p, n in zip(_group_params(groups), param_names(groups))}
    weight_stream = engines.weight_stream
    grad_norm = getattr(hooks, "grad_norm", local_grad_norm)
    if w.sharded and not w.gpu_blocks:
        groups, _ = split_groups(groups, _not_mine(weight_stream))
        grad_norm = ShardedNorm(partial(_owned_grads, weight_stream), device)
    dev_groups, block_groups = (
        split_groups(groups, w.streamed) if w.gpu_blocks else (groups, [])
    )
    if w.gpu_blocks and not dev_groups:
        raise ValueError(
            "swap with a GPU optimiser needs a trainable non-block parameter"
        )
    device_params = [
        p for g in dev_groups for p in g["params"] if id(p) not in w.streamed
    ]
    grads = None
    extra = {}
    if plan.grad_offload:
        grads = attach_grad_offload(engines, device_params, plan.pinned)
        extra = _offload_parts(
            grads, w.host_adamw, w.fused, engines.host_params(), by_id
        )
    if w.name in HOST_BUFFER_OPTIMIZERS:
        hyper = {**hyper, "pinned": plan.pinned}
    before = None if w.fused and w.ddp is not None else getattr(hooks, "before", None)
    bundle = build_bundle(
        dev_groups,
        device,
        w.name,
        hyper,
        schedule,
        steps,
        prec,
        grad_clip,
        fused_step=w.fused,
        grad_norm=grad_norm,
        agree=getattr(hooks, "agree", identity),
        before=_chain(
            extra.get("finish_unhooked"),
            before,
            engines.before_step,
            extra.get("before"),
        ),
        after=_chain(engines.after_step, extra.get("after")),
        reduce_loss=getattr(hooks, "reduce_loss", identity),
        grads=extra.get("grads"),
        grad_names=extra.get("grad_names"),
        reduce_step=getattr(hooks, "reduce_step", None),
    )
    wire = _wire_fused if w.fused else _wire_kept
    return wire(bundle, w, engines, grads, device_params, block_groups, hyper, hooks)


@dataclass(frozen=True)
class OptimWiring:
    """One plan's resolved optimiser wiring (`resolve_wiring`): the optimiser name,
    the in-backward step, where streamed blocks step, and the DDP role."""

    name: str
    fused: bool
    gpu_blocks: bool
    device_states: bool
    host_adamw: bool
    sharded: bool
    ddp: object | None
    streamed: frozenset[int]


def resolve_wiring(
    plan: MemoryPlan, engines: MemoryEngines, optimizer: str, hooks=None
) -> OptimWiring:
    """The wiring of OPTIMIZER under `plan` over `engines` and the layout's `hooks`;
    raises when GPU-stepped swap blocks lack the in-backward step."""
    name = optimizer_for(optimizer, plan)
    weight_stream = engines.weight_stream
    streamed = frozenset(id(p) for p in engines.host_params())
    gpu_blocks = bool(streamed) and plan.optim_state != "cpu"
    fused = bool(plan.fused_step)
    if gpu_blocks and not fused:
        raise ValueError(
            "swap blocks with a GPU optimiser step in backward: fused_step"
        )
    ddp = getattr(hooks, "ddp", None)
    return OptimWiring(
        name=name,
        fused=fused,
        gpu_blocks=gpu_blocks,
        device_states=plan.optim_state == "gpu",
        host_adamw=name == "cpu_adamw",
        sharded=ddp is not None
        and weight_stream is not None
        and weight_stream.owners is not None,
        ddp=ddp,
        streamed=streamed,
    )


def _wire_kept(
    bundle, w: OptimWiring, engines, grads, device_params, block_groups, hyper, hooks
):
    """Gradients kept until the step: the host optimiser reads offloaded sums, DDP sums
    streamed blocks and offloaded gradients, host-stepped blocks start each
    micro-batch."""
    if w.host_adamw and grads is not None:
        device_optimizer(bundle).use_host_grads(grads)
    final = None
    if w.ddp is not None:
        final = _ddp_kept(w.ddp, engines, grads, device_params)
    accumulate = getattr(hooks, "accumulate", None) or _null_accumulate
    if engines.host_stepped:
        accumulate = partial(_start_both, engines.weight_stream.start, accumulate)
    return bundle, _offload_start(grads, accumulate, final)


def _wire_fused(
    bundle, w: OptimWiring, engines, grads, device_params, block_groups, hyper, hooks
):
    """The step in backward: device parameters through gradient offload or FusedStep
    (DDP-reduced first), GPU-stepped swap blocks through StreamedAdamW."""
    ddp = w.ddp
    step = weak_method(bundle.step_param)
    if ddp is not None:
        step = ddp.reduced(device_params, step)
    if grads is not None:
        bundle.attach(
            grads if ddp is None else _OffloadThrough(grads, step), finishes=True
        )
    else:
        bundle.attach(attach_fused_step(engines, device_params, step))
    if w.gpu_blocks:
        _attach_blocks(bundle, w, engines.weight_stream, block_groups, hyper)
    if ddp is None:
        return bundle, bundle.start
    # ddp.finish after GradOffload.finish (index 0) when gradient offload is attached
    bundle.finishers.insert(0 if grads is None else 1, ddp.finish)
    return bundle, partial(_start_both, ddp.start, bundle.start)


def _attach_blocks(bundle, w: OptimWiring, weight_stream, block_groups, hyper) -> None:
    """StreamedAdamW over the swap blocks, its DDP reduction and the shared-region
    barrier."""
    blocks = StreamedAdamW(
        weight_stream,
        lr=hyper["lr"],
        betas=hyper["betas"],
        eps=hyper["eps"],
        weight_decay=0.0,
        device_states=w.device_states,
        groups=master_groups(block_groups),
    )
    bundle.attach_blocks(blocks)
    if w.ddp is not None:
        everyone = w.device_states or not w.sharded
        blocks.reduce = w.ddp.block_reduce(weight_stream.owners, everyone=everyone)
        w.ddp.watch(getattr(blocks, "opt", None))
        if not everyone:
            bundle.reduce_block_stats = sum_block_stats
    if weight_stream.arena is not None:
        bundle.finishers.append(
            partial(_host_barrier, blocks.synchronize, weight_stream.arena)
        )


def close_memory(engines: MemoryEngines, bundle) -> None:
    """End of a run: stop the host worker threads of the engines and the optimisers."""
    engines.close()
    bundle.close()


def _null_accumulate(last: bool):
    return contextlib.nullcontext()
