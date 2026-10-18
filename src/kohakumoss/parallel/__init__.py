"""Multi-device training: launcher, data parallel, FSDP2, pipelines, stage split,
balancing, per-rank report."""

from kohakumoss.parallel.ddp import GradReducer, broadcast_params, wrap_torch_ddp
from kohakumoss.parallel.launch import DistEnv, default_backend, spawn
from kohakumoss.parallel.pipeline.balance import measure_speed
from kohakumoss.parallel.pipeline.parts import BackboneParts, PartsCtx
from kohakumoss.parallel.pipeline.threaded import ThreadedPipeline

__all__ = [
    "BackboneParts",
    "DistEnv",
    "GradReducer",
    "PartsCtx",
    "ThreadedPipeline",
    "broadcast_params",
    "default_backend",
    "measure_speed",
    "spawn",
    "wrap_torch_ddp",
]
