"""Build the pipeline engines for PARALLEL 'threaded' (one process, several devices) and
'pipeline' (one process per stage), and restrict the trainable set to local stages."""

from functools import partial

import torch

from kohakumoss.model.opset import OpSet
from kohakumoss.parallel.parts import BackboneParts
from kohakumoss.parallel.pipeline import PipelineEngine, PipeShape
from kohakumoss.parallel.threaded import ThreadedPipeline, even_bounds
from kohakumoss.training.build.objective import build_loss
from kohakumoss.training.drivers.pipeline import PipelineObjective


def _opset_for(policy, kernels: dict, device):
    return OpSet.resolve(device, policy.compute, kernels)


def build_threaded(k: dict, built, policy):
    """ThreadedPipeline over DEVICES, each stage bound to its device's kernels."""
    parts = BackboneParts(built.backbone)
    devices = [torch.device("cuda", i) for i in k["DEVICES"]]
    schedule = k["PIPELINE_SCHEDULE"]
    return ThreadedPipeline(
        parts,
        devices,
        epilogue_modules=parts.epilogue_modules(),
        runner=built.backbone.runner,
        opset_for=partial(_opset_for, policy, k["KERNELS"]),
        schedule=schedule,
    )


def pipeline_objective(k: dict, built) -> PipelineObjective:
    """The last stage's loss: LOSS / LOSS_KWARGS, as TrainStep builds it (the scaler
    is set once the optimiser bundle exists)."""
    return PipelineObjective(build_loss(k, built), scaler=None)


def build_engine(k: dict, built, policy, rank: int, world: int, device):
    """(PipelineEngine for this rank, its objective) with static micro-batch shapes:
    MAX_TOKENS joint tokens and MAX_SAMPLES samples per micro-batch (both hard bounds of
    the planner under TOKEN_BUDGET='exact')."""
    parts = BackboneParts(built.backbone)
    cfg = built.backbone.cfg
    shape = PipeShape.from_knobs(k, dim=cfg.dim, rope_axes=cfg.rope_axes)
    per_rank = 1 if k["PIPELINE_SCHEDULE"] in ("gpipe", "1f1b") else 2
    bounds = even_bounds(cfg.depth, world * per_rank)
    objective = pipeline_objective(k, built)
    engine = PipelineEngine(
        parts,
        rank,
        world,
        device,
        shape,
        bounds,
        k["PIPELINE_SCHEDULE"],
        objective,
        policy.compute,
        runner=built.backbone.runner,
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
