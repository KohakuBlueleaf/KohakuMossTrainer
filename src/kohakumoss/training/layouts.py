"""PARALLEL as a registry slot: one `Layout` per kind holds every per-kind decision of
a run (process launch, data streams, loop hooks, pipeline stages, checkpoint parts,
driver, full-weight context), resolved once from `LAYOUTS` by the PARALLEL knob. A
dotted path names a Layout built elsewhere."""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from kohakumoss.parallel.fsdp import unsharded
from kohakumoss.parallel.pipeline.distributed import retained_backward
from kohakumoss.training.build.parallel import (
    DDP_BACKENDS,
    ParallelHooks,
    ddp_hooks,
    fsdp_hooks,
    pipeline_hooks,
    torch_ddp_hooks,
)
from kohakumoss.training.build.pipelines import (
    build_engine,
    build_threaded,
    local_names,
)
from kohakumoss.training.checks.parallel import (
    ddp_plan_problems,
    ddp_problems,
    fsdp_plan_problems,
    fsdp_problems,
    local_problems,
    pipeline_layout_problems,
    pipeline_plan_problems,
    threaded_plan_problems,
    threaded_problems,
)
from kohakumoss.training.context import RunContext
from kohakumoss.training.drivers import DRIVERS, Parts
from kohakumoss.training.drivers.common import ShardedOptimParts
from kohakumoss.training.drivers.fsdp import FsdpParts
from kohakumoss.training.drivers.pipeline import (
    PipelineDriver,
    PipelineParts,
    ThreadedDriver,
)
from kohakumoss_common.calls import nothing
from kohakumoss_common.registry import Registry


def _no_stages(ctx: RunContext, prec) -> tuple:
    return None, None, None, None


def _threaded_stages(ctx: RunContext, prec) -> tuple:
    return build_threaded(ctx.k, ctx.built, prec, ctx.device), None, None, None


def _pipeline_stages(ctx: RunContext, prec) -> tuple:
    engine, objective = build_engine(
        ctx.k, ctx.built, prec, ctx.rank, ctx.world, ctx.device
    )
    keep = {id(p) for p in engine.parameters()}
    return engine, objective, local_names(ctx.built.trainable, engine), keep


def _local_hooks(ctx: RunContext) -> ParallelHooks:
    return ParallelHooks()


def _ddp_hooks(ctx: RunContext) -> ParallelHooks:
    k = ctx.k
    match k["DDP_BACKEND"]:
        case "kohakumoss":
            return ddp_hooks(ctx.built, ctx.device, k["DDP_REDUCE_DTYPE"])
        case "torch":
            return torch_ddp_hooks(ctx.built, ctx.device, k["DDP_REDUCE_DTYPE"])
    raise SystemExit(f"DDP_BACKEND={k['DDP_BACKEND']!r} not in {DDP_BACKENDS}")


def _fsdp_hooks(ctx: RunContext) -> ParallelHooks:
    k = ctx.k
    return fsdp_hooks(
        ctx.built.backbone, ctx.world, k["FSDP_REDUCE"], k["FSDP_RESHARD"]
    )


def _pipeline_hooks(ctx: RunContext) -> ParallelHooks:
    return pipeline_hooks()


def _plain_parts(ctx: RunContext, args: tuple, names, extra) -> Parts:
    if getattr(ctx.built.engines.weight_stream, "owners", None) is not None:
        return ShardedOptimParts(*args, extra=extra)
    return Parts(*args, extra=extra)


def _fsdp_parts(ctx: RunContext, args: tuple, names, extra) -> Parts:
    return FsdpParts(*args, model=ctx.built.backbone, extra=extra)


def _pipeline_parts(ctx: RunContext, args: tuple, names, extra) -> Parts:
    return PipelineParts(
        *args, local_names=names, rank=ctx.rank, world=ctx.world, extra=extra
    )


def _loop_driver(ctx: RunContext, common: dict, accumulate, engine):
    return DRIVERS.get(ctx.k["DRIVER"])(**common, accumulate=accumulate)


def _threaded_driver(ctx: RunContext, common: dict, accumulate, engine):
    return ThreadedDriver(**common, pipe=engine)


def _pipeline_driver(ctx: RunContext, common: dict, accumulate, engine):
    return PipelineDriver(**common, engine=engine, world=ctx.world)


def _no_unshard(backbone):
    return contextlib.nullcontext()


def _pipeline_begin(k: dict) -> None:
    retained_backward(k["PIPELINE_SCHEDULE"])


def _no_problems(k: dict) -> list[str]:
    return []


def _no_plan_problems(k: dict, plan) -> list[str]:
    return []


@dataclass(frozen=True)
class Layout:
    """What one PARALLEL kind decides. `spawns`: one process per device;
    `shared_data`: every rank draws rank 0's batches, sigma and noise (one data rank);
    `pipelined`: a step is MICROBATCHES batches through a pipeline engine;
    `stage_rng`: checkpoints keep every stage device's RNG; `begin(k)` runs once per
    run before the build; `ctx` is the rank's training.context.RunContext:
    `hooks(ctx)`, `stages(ctx, prec)` -> (engine, objective, local trainable names,
    local parameter ids), `parts(ctx, args, names, extra)`, `driver(ctx, common,
    accumulate, engine)`, `full_weights(backbone)` -> the context in
    which every rank holds full weights, `problems(k)` / `plan_problems(k, plan)` ->
    the knob / memory-plan problems of this layout (the knob checks call the selected
    layout's hooks instead of branching on kind names)."""

    name: str
    spawns: bool
    hooks: Callable
    problems: Callable[[dict], list[str]] = _no_problems
    plan_problems: Callable = _no_plan_problems
    begin: Callable = nothing
    parts: Callable = _plain_parts
    stages: Callable = _no_stages
    driver: Callable = _loop_driver
    full_weights: Callable = _no_unshard
    shared_data: bool = False
    pipelined: bool = False
    stage_rng: bool = False

    def data_rank(self, rank: int) -> int:
        """The rank whose data, sigma and noise streams this rank draws."""
        return 0 if self.shared_data else rank

    def data_world(self, world: int) -> int:
        """Ranks the data is split over."""
        return 1 if self.shared_data else world


LAYOUTS = Registry("parallel layout")
for layout in (
    Layout("none", spawns=False, hooks=_local_hooks, problems=local_problems),
    Layout(
        "ddp",
        spawns=True,
        hooks=_ddp_hooks,
        problems=ddp_problems,
        plan_problems=ddp_plan_problems,
    ),
    Layout(
        "fsdp",
        spawns=True,
        hooks=_fsdp_hooks,
        problems=fsdp_problems,
        plan_problems=fsdp_plan_problems,
        parts=_fsdp_parts,
        full_weights=unsharded,
    ),
    Layout(
        "pipeline",
        spawns=True,
        hooks=_pipeline_hooks,
        problems=pipeline_layout_problems,
        plan_problems=pipeline_plan_problems,
        begin=_pipeline_begin,
        parts=_pipeline_parts,
        stages=_pipeline_stages,
        driver=_pipeline_driver,
        shared_data=True,
        pipelined=True,
    ),
    Layout(
        "threaded",
        spawns=False,
        hooks=_local_hooks,
        problems=threaded_problems,
        plan_problems=threaded_plan_problems,
        stages=_threaded_stages,
        driver=_threaded_driver,
        pipelined=True,
        stage_rng=True,
    ),
):
    LAYOUTS.add(layout.name, layout)


def layout_of(k: dict) -> Layout:
    """The Layout PARALLEL names (a registered kind or a dotted path to a Layout)."""
    return LAYOUTS.get(k["PARALLEL"])
