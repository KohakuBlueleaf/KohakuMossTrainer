"""Build the pipeline engines for PARALLEL 'threaded' (one process, several devices) and
'pipeline' (one process per stage), and restrict the trainable set to local stages."""

from functools import partial

import torch

from kohakumoss.model.opset import OpSet
from kohakumoss.parallel.pipeline.distributed import (
    PipelineEngine,
    PipeShape,
    TokenLoss,
    stages_per_rank,
)
from kohakumoss.parallel.pipeline.parts import BackboneParts
from kohakumoss.parallel.pipeline.split import stage_bounds
from kohakumoss.parallel.pipeline.threaded import ThreadedPipeline
from kohakumoss.precision import autocast_factory
from kohakumoss.training.build.objective import build_loss


def _opset_for(prec, kernels: dict, device):
    return OpSet.resolve(device, prec.compute, kernels)


def build_threaded(k: dict, built, prec, device: torch.device):
    """ThreadedPipeline over DEVICES (of the training device's type), each stage bound
    to its device's kernels."""
    parts = BackboneParts(built.backbone)
    devices = [torch.device(device.type, i) for i in k["DEVICES"]]
    schedule = k["PIPELINE_SCHEDULE"]
    return ThreadedPipeline(
        parts,
        devices,
        epilogue_modules=parts.epilogue_modules(),
        runner=built.backbone.runner,
        opset_for=partial(_opset_for, prec, k["KERNELS"]),
        schedule=schedule,
    )


def pipeline_objective(k: dict, built) -> TokenLoss:
    """The last stage's loss: LOSS / LOSS_KWARGS, as TrainStep builds it (the scaler
    is set once the optimiser bundle exists)."""
    return TokenLoss(build_loss(k, built))


def build_engine(k: dict, built, prec, rank: int, world: int, device):
    """(PipelineEngine for this rank, its objective) with static micro-batch shapes
    (MAX_TOKENS joint tokens and MAX_SAMPLES samples per micro-batch, hard bounds of
    the planner under TOKEN_BUDGET='exact') and the PIPELINE_SPLIT stage ranges."""
    parts = BackboneParts(built.backbone)
    cfg = built.backbone.cfg
    shape = PipeShape.from_knobs(k, dim=cfg.dim, rope_axes=cfg.rope_axes)
    schedule = k["PIPELINE_SCHEDULE"]
    per_rank = stages_per_rank(schedule)
    runner = built.backbone.runner
    bounds = stage_bounds(
        k["PIPELINE_SPLIT"],
        parts,
        shape,
        schedule,
        rank,
        world,
        device,
        autocast_factory(prec, device),
        runner,
        per_rank,
    )
    objective = pipeline_objective(k, built)
    engine = PipelineEngine(
        parts,
        rank,
        world,
        device,
        shape,
        bounds,
        schedule,
        objective,
        prec.compute,
        runner=runner,
        per_rank=per_rank,
    )
    return engine, objective


def release_nonlocal(backbone, keep: set[int]) -> None:
    """Move every block and top-level module with no parameter in `keep` (this rank's
    stages) to host memory."""
    modules = list(backbone.layers) + [
        m for m in backbone.children() if m is not backbone.layers
    ]
    for module in modules:
        params = list(module.parameters())
        if params and not any(id(p) in keep for p in params):
            module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def local_names(trainable, engine) -> set[str]:
    """Trainable entries whose parameter lives in this rank's stages."""
    local = {id(p) for p in engine.parameters()}
    return {name for name, p in trainable.named_trainable() if id(p) in local}


def filter_groups(groups: list[dict], keep: set[int]) -> list[dict]:
    """Optimiser groups restricted to parameters in `keep` (empty groups dropped)."""
    out = []
    for g in groups:
        idx = [i for i, p in enumerate(g["params"]) if id(p) in keep]
        if not idx:
            continue
        ng = dict(g)
        for key in ("params", "names", "splits"):
            if key in g:
                ng[key] = [g[key][i] for i in idx]
        out.append(ng)
    return out
