"""`run(globals())`: validate every knob, build in the fixed order, train, export."""

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.logging import build_logger
from kohakumoss.registry import import_object
from kohakumoss.training.build import integration as I
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
)
from kohakumoss.training.build.model import build_model, export
from kohakumoss.training.build.parallel import build_hooks
from kohakumoss.training.build.pipelines import (
    build_engine,
    build_threaded,
    local_names,
)
from kohakumoss.training.callbacks.evaluate import FixedSigma
from kohakumoss.training.drivers import Parts
from kohakumoss.training.drivers.pipeline import PipelineParts
from kohakumoss.training.knobs import check
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.training.state import TrainState
from kohakumoss.types.plans import ParallelPlan


def _device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


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
        return import_object(I.PARALLEL_SPAWN)(
            worker_env, plan, k, device_type=_device_type()
        )
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


def _snapshot(k: dict) -> dict:
    return json.loads(json.dumps(k, default=str))


def worker(k: dict, rank: int = 0, world: int = 1, device=None) -> TrainState:
    """One rank of a run."""
    is_main = rank == 0
    device = device or _device(k, rank)
    torch.manual_seed(k["SEED"])  # identical model + adapter init on every rank
    output_dir = str(Path(k["OUTPUT_DIR"]) / (k["RUN_NAME"] or "run"))
    if is_main:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "config.json").write_text(
            json.dumps(_snapshot(k), indent=2)
        )
    import_object(I.KERNEL_TUNE_MODE)(k["TUNE"])
    if device.type == "cuda":  # MAGMA's batched inverse (OFT Cayley) faults on sm86
        torch.backends.cuda.preferred_linalg_library("cusolver")
    policy = build_policy(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    built = build_model(k, device, policy)
    engine, objective, names, keep = _stages(k, built, policy, rank, world, device)
    # prewarm before the gradient-sync hooks: its scratch backward must not reduce
    warm = prewarm_devices(k, device)
    build_prewarm(k, built.backbone, policy.compute)(warm)
    build_adapter_prewarm(k, built, policy.compute)(warm)
    torch.manual_seed(k["SEED"] + rank)  # per-rank stream for everything after build
    hooks = build_hooks(k["PARALLEL"], built, policy.compute, world)
    _compile(built.backbone, k["COMPILE"])
    normalizer = TokenNormalizer(hooks.reduce, hooks.world_factor)
    autocast = autocast_factory(policy, device)
    step = build_step(k, built, normalizer, autocast, device)
    bundle, accumulate = build_optim(k, built, device, policy, hooks, keep)
    if objective is not None:
        objective.scaler = bundle.scaler
    ema = build_ema_for(k, built, device, names)
    data_cfg = import_object(I.DATA_CONFIG).from_knobs(k)
    data_rank, data_world = (0, 1) if k["PARALLEL"] == "pipeline" else (rank, world)
    loader = import_object(I.DATA_LOADER)(data_cfg, rank=data_rank, world=data_world)
    writer = build_writer(k, loader, writes=data_rank == rank)
    batcher = build_batcher(k, built, device, writer)
    val_batches = []
    if k["VALIDATE_EVERY"] > 0:
        val_cfg = import_object(I.DATA_CONFIG).from_knobs(
            {**k, "DATASETS": k["VALIDATION"]["datasets"]}
        )
        val_loader = import_object(I.DATA_LOADER)(val_cfg, rank=rank, world=world)
        val_batches = fixed_batches(
            val_loader,
            build_batcher(k, built, device),
            int(k["VALIDATION"].get("batches", 4)),
        )
    logger = build_logger(
        k["LOGGER"],
        output_dir,
        k["RUN_NAME"] or "run",
        k.get("PROJECT", "kohakumoss"),
        _snapshot(k),
        is_main,
    )

    def make_step(sampler: FixedSigma):
        return build_step(k, built, normalizer, autocast, device, sampler=sampler)

    ckpt = build_checkpoints(k, output_dir, ema, is_main, writer)
    parts_args = (built.trainable, ema, bundle, step, loader)
    if k["PARALLEL"] == "pipeline":
        parts = PipelineParts(
            *parts_args,
            local_names=names,
            rank=rank,
            world=world,
            extra=lambda: {"world": world},
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
    if is_main:
        meta = {"kohakumoss_step": str(state.step), "kohakumoss_algo": str(k["ALGO"])}
        for path in export(k, built, output_dir, meta):
            print(f"exported {path}", file=sys.stderr)
    logger.close()
    return state
