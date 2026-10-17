"""`run(globals())`: validate every knob, build in the fixed order, train, export."""

import contextlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.logging import build_logger
from kohakumoss.parallel.fsdp import unsharded
from kohakumoss.parallel.launch import spawn
from kohakumoss.training.build.kernels import (
    build_adapter_prewarm,
    build_prewarm,
    prewarm_devices,
)
from kohakumoss.training.build.loop import (
    build_batcher,
    build_callbacks,
    build_checkpoints,
    build_driver,
    build_ema_for,
    build_optim,
    build_step,
    build_writer,
    fixed_batches,
    validation_data_config,
)
from kohakumoss.training.build.model import build_model, export
from kohakumoss.training.build.parallel import build_hooks
from kohakumoss.training.build.pipelines import (
    build_engine,
    build_threaded,
    local_names,
    release_nonlocal,
)
from kohakumoss.training.callbacks.evaluate import FixedSigma
from kohakumoss.training.drivers import Parts
from kohakumoss.training.drivers.fsdp import FsdpParts
from kohakumoss.training.drivers.pipeline import PipelineParts
from kohakumoss.training.knobs import check, default_device_type, run_name
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import LayoutTransform
from kohakumoss.types.plans import ParallelPlan
from kohakumoss_data import DataConfig, build_loader
from kohakumoss_kernels.tuning.tuner import set_mode as set_tune_mode

_device_type = default_device_type


def run(g: dict):
    """Entry point of scripts/train/moss.py; multi-device kinds spawn one process each."""
    k = check(g)
    if k["PARALLEL"] not in ("none", "threaded") and not dist.is_initialized():
        plan = ParallelPlan(
            kind=k["PARALLEL"],
            backend=k["DIST_BACKEND"],
            devices=tuple(k["DEVICES"]),
            schedule=k["PIPELINE_SCHEDULE"],
            microbatches=k["MICROBATCHES"],
        )
        return spawn(worker_env, plan, k, device_type=_device_type())
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return worker(k, rank, world)


def worker_env(env, k: dict) -> TrainState:
    """Spawned-process entry: `env` is kohakumoss.parallel.launch.DistEnv."""
    return worker(k, env.rank, env.world, env.device)


def _device(k: dict, rank: int) -> torch.device:
    index = k["DEVICES"][rank % len(k["DEVICES"])]
    match _device_type():
        case "cuda":
            torch.cuda.set_device(index)
            return torch.device("cuda", index)
        case "xpu":
            return torch.device("xpu", index)
        case _:
            return torch.device("cpu")


def _compile(backbone, mode: str) -> None:
    match mode:
        case "block":
            for block in backbone.layers:
                block.compile(dynamic=True)
        case "model":
            backbone.compile(dynamic=True)


def _stages(k: dict, built, policy, rank: int, world: int, device):
    """(engine, objective, local trainable names, local param ids) for the pipelined
    PARALLEL kinds; (None, None, None, None) otherwise (everything is local)."""
    match k["PARALLEL"]:
        case "threaded":
            return build_threaded(k, built, policy), None, None, None
        case "pipeline":
            engine, objective = build_engine(k, built, policy, rank, world, device)
            keep = {id(p) for p in engine.parameters()}
            return engine, objective, local_names(built.trainable, engine), keep
    return None, None, None, None


def stream_rank(kind: str, rank: int) -> int:
    """The rank whose data, sigma and noise streams this rank draws: its own under data
    parallelism, rank 0's under `pipeline` (every stage sees the same micro-batches)."""
    return 0 if kind == "pipeline" else rank


def _snapshot(k: dict) -> dict:
    return json.loads(json.dumps(k, default=str))


def deterministic_mode() -> None:
    """KOHAKUMOSS_DETERMINISTIC=1 (set before the process imports the kernels):
    torch's deterministic algorithms, a fixed cuBLAS workspace, no cuDNN autotune;
    the kernels read the same variable. See docs/concepts/training.md."""
    if os.environ.get("KOHAKUMOSS_DETERMINISTIC") != "1":
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def worker(k: dict, rank: int = 0, world: int = 1, device=None) -> TrainState:
    """One rank of a run."""
    is_main = rank == 0
    deterministic_mode()
    device = device or _device(k, rank)
    torch.manual_seed(k["SEED"])  # identical model + adapter init on every rank
    output_dir = str(Path(k["OUTPUT_DIR"]) / run_name(k))
    if is_main:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "config.json").write_text(
            json.dumps(_snapshot(k), indent=2)
        )
    set_tune_mode(k["TUNE"])
    if device.type == "cuda":  # cuSOLVER linear algebra, see docs/concepts/training.md
        torch.backends.cuda.preferred_linalg_library("cusolver")
    policy = build_policy(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    built = build_model(k, device, policy, world)
    engine, objective, names, keep = _stages(k, built, policy, rank, world, device)
    if keep is not None:
        release_nonlocal(built.backbone, keep)
    # kernel and adapter prewarm, before the gradient-sync hooks are installed
    warm = prewarm_devices(k, device)
    build_prewarm(k, built.backbone, policy.compute)(warm)
    build_adapter_prewarm(k, built, policy.compute)(warm)
    torch.manual_seed(k["SEED"] + rank)  # per-rank stream for everything after build
    hooks = build_hooks(k["PARALLEL"], built, policy.compute, world)
    _compile(built.backbone, built.plan.compile)
    normalizer = TokenNormalizer(hooks.reduce, hooks.world_factor)
    autocast = autocast_factory(policy, device)
    data_rank = stream_rank(k["PARALLEL"], rank)
    data_world = 1 if k["PARALLEL"] == "pipeline" else world
    step = build_step(k, built, normalizer, autocast, device, stream_rank=data_rank)
    bundle, accumulate = build_optim(k, built, device, policy, hooks, keep)
    if objective is not None:
        objective.scaler = bundle.scaler
    ema = build_ema_for(k, built, device, names)
    data_cfg = DataConfig.from_knobs(k)
    layout = LayoutTransform(built.backbone.cfg.rope_axes)
    loader = build_loader(data_cfg, rank=data_rank, world=data_world, transform=layout)
    writer = build_writer(k, loader, writes=data_rank == rank)
    batcher = build_batcher(k, built, device, writer, loader.dataset)
    val_batches = []
    if k["VALIDATE_EVERY"] > 0:
        val_loader = build_loader(
            validation_data_config(k), rank=rank, world=world, transform=layout
        )
        val_batches = fixed_batches(
            val_loader,
            build_batcher(k, built, device, dataset=val_loader.dataset),
            int(k["VALIDATION"].get("batches", 4)),
        )
    logger = build_logger(
        k["LOGGER"],
        output_dir,
        run_name(k),
        k.get("PROJECT", "kohakumoss"),
        _snapshot(k),
        is_main,
    )

    def make_step(sampler: FixedSigma):
        return build_step(
            k, built, normalizer, autocast, device, sampler, stream_rank=data_rank
        )

    ckpt = build_checkpoints(k, output_dir, ema, is_main, writer, rank, warm)
    parts_args = (built.trainable, ema, bundle, step, loader)
    if k["PARALLEL"] == "pipeline":
        parts = PipelineParts(
            *parts_args,
            local_names=names,
            rank=rank,
            world=world,
            extra=lambda: {"world": world},
        )
    elif k["PARALLEL"] == "fsdp":
        parts = FsdpParts(
            *parts_args, model=built.backbone, extra=lambda: {"world": world}
        )
    else:
        parts = Parts(*parts_args, extra=lambda: {"world": world})
    runner = build_callbacks(
        k,
        built,
        bundle,
        logger,
        output_dir,
        device,
        is_main,
        make_step,
        val_batches,
        ckpt,
        parts,
        reduce_loss=hooks.reduce_loss,
    )
    state = TrainState()
    resume = ckpt.resolve(k["RESUME"])
    if resume is not None:
        parts.restore(ckpt.load(resume), state)
    driver = build_driver(
        k, batcher, normalizer, device, runner, ckpt, parts, accumulate, engine, world
    )
    driver.run(step, loader, bundle, ema, state)
    if writer is not None:
        writer.close()
    with full_weights(k["PARALLEL"], built.backbone):
        if is_main:
            meta = {
                "kohakumoss_step": str(state.step),
                "kohakumoss_algo": str(k["ALGO"]),
            }
            for path in export(k, built, output_dir, meta):
                print(f"exported {path}", file=sys.stderr)
    logger.close()
    return state


def full_weights(kind: str, backbone):
    """Context in which every rank's backbone holds full weights (FSDP2: unsharded on
    every rank; otherwise nothing to do)."""
    if kind == "fsdp":
        return unsharded(backbone)
    return contextlib.nullcontext()
