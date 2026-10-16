"""Build the pipeline engines for PARALLEL 'threaded' (one process, several devices) and
'pipeline' (one process per stage), and restrict the trainable set to local stages."""

from functools import partial

import torch

from kohakumoss.model.opset import OpSet
from kohakumoss.registry import import_object
from kohakumoss.training.build import integration as I
from kohakumoss.training.drivers.pipeline import PipelineObjective

BACKBONE_PARTS = "kohakumoss.parallel.parts.BackboneParts"
EVEN_BOUNDS = "kohakumoss.parallel.threaded.even_bounds"
PIPE_SHAPE = "kohakumoss.parallel.pipeline.PipeShape"


def _opset_for(policy, kernels: dict, device):
    return OpSet.resolve(device, policy.compute, kernels)


def build_threaded(k: dict, built, policy):
    """ThreadedPipeline over DEVICES, each stage bound to its device's kernels."""
    parts = import_object(BACKBONE_PARTS)(built.backbone)
    devices = [torch.device("cuda", i) for i in k["DEVICES"]]
    schedule = k["PIPELINE_SCHEDULE"] if k["PIPELINE_SCHEDULE"] in ("gpipe", "async") else "gpipe"
    return import_object(I.THREADED_PIPELINE)(
        parts,
        devices,
        epilogue_modules=parts.epilogue_modules(),
        runner=built.backbone.runner,
        opset_for=partial(_opset_for, policy, k["KERNELS"]),
        schedule=schedule,
    )


def build_engine(k: dict, built, policy, rank: int, world: int, device):
    """(PipelineEngine for this rank, its objective) with static micro-batch shapes:
    MAX_TOKENS joint tokens and MAX_SAMPLES samples per micro-batch (both hard bounds of
    the planner under TOKEN_BUDGET='exact')."""
    parts = import_object(BACKBONE_PARTS)(built.backbone)
    cfg = built.backbone.cfg
    shape = import_object(PIPE_SHAPE)(
        tokens=k["MAX_TOKENS"],
        max_samples=k["MAX_SAMPLES"],
        dim=cfg.dim,
        microbatches=k["MICROBATCHES"],
    )
    per_rank = 1 if k["PIPELINE_SCHEDULE"] in ("gpipe", "1f1b") else 2
    bounds = import_object(EVEN_BOUNDS)(cfg.depth, world * per_rank)
    objective = PipelineObjective(built.backbone.ops.token_mse, scaler=None)
    engine = import_object(I.PIPELINE_ENGINE)(
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
