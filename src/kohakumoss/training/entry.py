"""`run(globals())`: validate every knob, build in the fixed order, train, export."""

import json
import multiprocessing
from functools import partial
from pathlib import Path

import torch

from kohakumoss.adapters.lycoris.tuning import (
    restore_lycoris_tuning,
    set_lycoris_tuning,
)
from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.logging import build_logger, fresh_run
from kohakumoss.logging.log import (
    add_log_file,
    bars,
    get_logger,
    hl,
    log_object,
    setup,
    timed,
)
from kohakumoss.memory.device_use import build_device_use
from kohakumoss.memory.timeline import build_timeline
from kohakumoss.model import resolve_config
from kohakumoss.offload.accel import configure_allocator, release_cache
from kohakumoss.parallel.launch import (
    gate_name,
    ranks_meet,
    release_ranks,
    set_gate_timeout,
    spawn,
    wait_for_rank0,
)
from kohakumoss.parallel.telemetry.report import build_rank_callbacks, rank_layout
from kohakumoss.precision import autocast_factory, build_precision
from kohakumoss.training import shutdown
from kohakumoss.training.build.kernels import (
    adapter_prewarm_devices,
    attach_text_prewarm,
    build_adapter_prewarm,
    build_prewarm,
)
from kohakumoss.training.build.loop import (
    build_ahead,
    build_batcher,
    build_callbacks,
    build_checkpoints,
    build_driver,
    build_ema_for,
    build_latent_ahead,
    build_optim,
    build_recorder,
    build_step,
    build_writer,
    fixed_batches,
    resumed_from,
    validation_data_config,
)
from kohakumoss.training.build.model import (
    build_model,
    data_knobs,
    export,
    placement_lines,
)
from kohakumoss.training.build.pipelines import release_nonlocal
from kohakumoss.training.callbacks.evaluate import FixedSigma
from kohakumoss.training.context import RunContext
from kohakumoss.training.drivers import Parts
from kohakumoss.training.encode import EncodeFeed, RankFeed, StreamedLoader, pin_batches
from kohakumoss.training.exports import (
    export_bytes,
    export_files,
    fused_qkv,
    log_export_plan,
)
from kohakumoss.training.knobs import check, default_device_type, is_full, run_name
from kohakumoss.training.layouts import Layout, layout_of
from kohakumoss.training.memory.optimizers import close_memory
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import LayoutTransform
from kohakumoss.tuning.prewarm import prewarm_devices, release_workspaces, tunes
from kohakumoss.types.plans import ParallelPlan
from kohakumoss_common.calls import nothing
from kohakumoss_data import DataConfig, MossDataset, build_loader
from kohakumoss_kernels import compile_setup
from kohakumoss_kernels.dist import dist_rank, initialized
from kohakumoss_kernels.run_state import reset_run_state
from kohakumoss_kernels.tuning import decisions, window
from kohakumoss_kernels.tuning.tuner import clear_caps
from kohakumoss_kernels.tuning.tuner import set_mode as set_tune_mode

_device_type = default_device_type
log = get_logger(__name__)


def run(g: dict):
    """Entry point of scripts/train/moss.py: the run's TrainState in process (PARALLEL
    'none' / 'threaded'); the process-group kinds spawn one process each and return
    None."""
    shutdown.install()
    setup()
    k = check(g)
    if k["ENCODE_DEVICES"] and not initialized():
        return run_fed(k)
    if layout_of(k).spawns and not initialized():
        return spawn(worker_env, parallel_plan(k), k, device_type=_device_type())
    rank, world = dist_rank()
    return worker(k, rank, world)


def parallel_plan(k: dict) -> ParallelPlan:
    """The ParallelPlan of the process-group layouts."""
    return ParallelPlan(
        kind=k["PARALLEL"],
        backend=k["DIST_BACKEND"],
        devices=tuple(k["DEVICES"]),
        schedule=k["PIPELINE_SCHEDULE"],
        microbatches=k["MICROBATCHES"],
        timeout_s=float(k["DIST_TIMEOUT"]),
        bind=k["DIST_BIND"],
    )


def run_fed(k: dict) -> None:
    """A run with ENCODE_DEVICES: this process runs the encode feed (data loading,
    the encoder processes, the batch streams) and the training runs in processes of
    its own: one per device for the process-group layouts, else one."""
    layout = layout_of(k)
    world = len(k["DEVICES"]) if layout.spawns else 1
    feed = EncodeFeed(k, layout, world, _device_type())
    feed.start()
    try:
        if layout.spawns:
            spawn(
                worker_env,
                parallel_plan(k),
                k,
                feed.handles,
                device_type=_device_type(),
            )
        else:
            run_child(worker_child, k, feed.handles[0])
    finally:
        feed.close()


def run_child(fn, *args) -> None:
    """`fn(*args)` in one spawned child process, joined; a failing child raises."""
    proc = multiprocessing.get_context("spawn").Process(target=fn, args=args)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"the training process exited with code {proc.exitcode}")


def worker_env(env, k: dict, feeds: list | None = None) -> TrainState:
    """Spawned-process entry: `env` is kohakumoss.parallel.launch.DistEnv; `feeds`
    holds every rank's end of the encode feed (ENCODE_DEVICES)."""
    setup(rank=env.rank, world=env.world)
    feed = None if feeds is None else feeds[env.rank]
    return worker(k, env.rank, env.world, env.device, feed)


def worker_child(k: dict, feed: RankFeed) -> None:
    """The training process of a single-process layout under the encode feed."""
    shutdown.install()
    setup()
    worker(k, 0, 1, None, feed)


def writer_first(rank: int, world: int, build):
    """`build(writer=...)` with one catalog writer: rank 0 builds (scans, writes the
    catalog files) before the other ranks build as readers; they wait on the store,
    not in a collective, so a long scan does not hit DIST_TIMEOUT."""
    if world == 1:
        return build(writer=True)
    name = gate_name("catalog")
    if rank == 0:
        out = build(writer=True)
        release_ranks(name)
        return out
    wait_for_rank0(name)
    return build(writer=False)


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
    compile_setup.apply()
    match mode:
        case "block":
            for block in backbone.layers:
                block.compile(dynamic=True)
        case "model":
            backbone.compile(dynamic=True)


def _snapshot(k: dict) -> dict:
    return json.loads(json.dumps(k, default=str))


def _start(k: dict, layout: Layout, rank: int, device: torch.device):
    """(output dir, precision, the LyCORIS tuning switch to restore): the run directory
    and its files on rank 0, the per-run process state, the run line."""
    allocator = configure_allocator(device)
    compile_pool = compile_setup.apply()
    torch.manual_seed(k["SEED"])
    output_dir = str(Path(k["OUTPUT_DIR"]) / run_name(k))
    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if resumed_from(k, output_dir) is None:
            fresh_run(output_dir)
        (Path(output_dir) / "config.json").write_text(
            json.dumps(_snapshot(k), indent=2), encoding="utf-8"
        )
        add_log_file(Path(output_dir) / "log.txt")
    reset_run_state()
    set_gate_timeout(k["DIST_GATE_TIMEOUT"])
    set_tune_mode(k["TUNE"])
    lycoris_tuning = set_lycoris_tuning(tunes(k))
    layout.begin(k)
    if device.type == "cuda":  # cuSOLVER linear algebra, see docs/concepts/training.md
        torch.backends.cuda.preferred_linalg_library("cusolver")
    prec = build_precision(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    log.info(
        f"run {hl(run_name(k))}: {hl(k['ALGO'])} {hl(k['PRECISION'])} on {device}, "
        f"{hl(k['STEPS'])} steps, output {hl(output_dir, 'path')}, allocator "
        f"{hl(allocator)}, compile pool: {hl(compile_pool)}"
    )
    return output_dir, prec, lycoris_tuning


def _data(k: dict, layout: Layout, rank: int, world: int, device):
    """(data knobs, token-layout transform, training loader): the catalog scan and
    batch plan over `layout`'s data ranks (rank 0 writes the catalog first)."""
    data_rank = layout.data_rank(rank)
    data_world = layout.data_world(world)
    kd = data_knobs(k, device)
    data_cfg = DataConfig.from_knobs(kd)
    arch = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    tokens = LayoutTransform(arch.rope_axes)
    with timed(log, "building the data loader (catalog scan, batch plan)"):
        loader = writer_first(
            rank,
            world,
            partial(build_loader, data_cfg, data_rank, data_world, tokens),
        )
    return kd, tokens, loader


def _fed_data(k: dict, layout: Layout, rank: int, world: int, device, feed: RankFeed):
    """(data knobs, token-layout transform, StreamedLoader) of a rank under the encode
    feed: the feed's data knobs, the dataset read only (the feed wrote the catalog),
    batches from the feed."""
    kd = feed.knobs
    cfg = DataConfig.from_knobs(kd)
    arch = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    tokens = LayoutTransform(arch.rope_axes)
    dataset = MossDataset(cfg, writer=False)
    devices = ", ".join(str(i) for i in k["ENCODE_DEVICES"])
    pin = pin_batches(device, cfg.pin_memory)
    return kd, tokens, StreamedLoader(feed, dataset, pin, devices)


def _prewarm(k: dict, built, prec, device, dataset, device_use) -> list:
    """The kernel prewarm (weightless twin, segment counts from the dataset) and the
    adapter prewarm, before the stage split and the gradient-sync hooks; the
    prewarmed devices."""
    warm = prewarm_devices(device)
    if not (warm and tunes(k)):
        log.info(f"no kernel prewarm (TUNE={k['TUNE']!r}, devices {warm or 'cpu'})")
        return warm
    with timed(log, f"kernel prewarm on {warm} (fwd + bwd, {prec.compute})"):
        build_prewarm(k, built, prec, device, dataset)(warm)
    decisions.flush("after the kernel prewarm")
    release_workspaces(warm)
    adapters = adapter_prewarm_devices(k, device)
    with timed(log, f"adapter prewarm on {adapters}"):
        build_adapter_prewarm(k, built, prec.compute)(adapters)
    release_workspaces(adapters)
    device_use.log(log, "after prewarm")
    return warm


def worker(
    k: dict, rank: int = 0, world: int = 1, device=None, feed: RankFeed | None = None
) -> TrainState:
    """One rank of a run; `feed`: the rank's end of the encode feed (ENCODE_DEVICES)."""
    is_main = rank == 0
    layout = layout_of(k)
    device = device or _device(k, rank)
    output_dir, prec, lycoris_tuning = _start(k, layout, rank, device)
    device_use = build_device_use(device)
    data_rank = layout.data_rank(rank)
    data_world = layout.data_world(world)
    if feed is None:
        kd, tokens, loader = _data(k, layout, rank, world, device)
    else:
        kd, tokens, loader = _fed_data(k, layout, rank, world, device, feed)
    ranks_meet("data")
    torch.manual_seed(k["SEED"])
    streamed = None if feed is None else loader
    with timed(log, "building model, adapters and encoders"):
        built = build_model(
            k, device, prec, world, loader.dataset.text_tokens_max(), streamed
        )
    log_object(log, "training memory plan", built.plan)
    for message, fields in placement_lines(k, built, device):
        log.info(message, **fields)
    files = export_files(k, fused_qkv(k))
    log_export_plan(k, files, export_bytes(files, built.trainable, built.backbone))
    release_cache(device)
    device_use.log(log, "after build")
    warm = _prewarm(k, built, prec, device, loader.dataset, device_use)
    ranks_meet("prewarmed")
    ctx = RunContext(
        k, built, layout, device, rank, world, data_world, is_main, output_dir
    )
    engine, objective, names, keep = layout.stages(ctx, prec)
    if keep is not None:
        release_nonlocal(built.backbone, keep)
    torch.manual_seed(k["SEED"] + rank)  # per-rank stream for everything after build
    hooks = layout.hooks(ctx)
    _compile(built.backbone, built.plan.compile)
    normalizer = TokenNormalizer(hooks.reduce, hooks.world_factor)
    autocast = autocast_factory(prec, device)
    stage_devices = [torch.device(device.type, i) for i in k["DEVICES"]]
    share, traced, sources = rank_layout(
        layout.name, engine, built.backbone.cfg.depth, stage_devices
    )
    recorder = build_recorder(k, built)
    step = build_step(
        k,
        built,
        normalizer,
        autocast,
        device,
        data_rank=data_rank,
        forward=hooks.forward,
        recorder=recorder,
    )
    bundle, accumulate = build_optim(k, built, device, prec, hooks, keep)
    if objective is not None:
        objective.scaler = bundle.scaler
    ema = build_ema_for(k, built, bundle, names, rank)
    writes = data_rank == rank and built.feed is None
    writer = build_writer(k, loader, writes=writes, rank=rank)
    loader, batcher = _inputs(k, built, device, loader, writer)
    timeline = build_timeline(device)
    for enc in (built.text_encoder, built.vae):
        if hasattr(enc, "set_timeline"):
            enc.set_timeline(timeline)
    val_batches = _validation_batches(k, built, device, kd, tokens, rank, world)
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
            k,
            built,
            normalizer,
            autocast,
            device,
            sampler,
            data_rank=data_rank,
            penalty=False,
        )

    rng_devices = stage_devices if layout.stage_rng else warm
    ckpt = build_checkpoints(k, output_dir, ema, is_main, writer, rank, rng_devices)
    parts = _parts(ctx, (ema, bundle, step, loader), names)
    ckpt.after_save = checkpoint_exports(k, layout, built, parts, is_main)
    rank_cbs = _rank_callbacks(ctx, [hooks.timing, *sources], share, traced)
    runner = build_callbacks(
        ctx,
        bundle,
        logger,
        make_step,
        val_batches,
        ckpt,
        parts,
        reduce_loss=hooks.reduce_loss,
        autocast=autocast,
        timeline=timeline,
        device_use=device_use,
        rank_callbacks=rank_cbs,
        recorder=recorder,
    )
    state = TrainState()
    resume = ckpt.resolve(k["RESUME"])
    if resume is not None:
        log.info(f"resuming from {resume}")
        parts.restore(ckpt.load(resume), state)
        built.engines.after_restore()
        logger.resume_at(state.step)
    driver = build_driver(
        ctx, batcher, normalizer, runner, ckpt, parts, accumulate, engine
    )
    log.info(f"training from step {state.step} to {k['STEPS']}")
    try:
        with bars():
            driver.run(step, loader, bundle, ema, state)
        if writer is not None:
            writer.close()
            writer = None
        _export(k, layout, built, parts, state, output_dir, is_main)
    finally:
        close_memory(built.engines, bundle)
        close_quietly(writer)
        logger.close()
        restore_lycoris_tuning(lycoris_tuning)
        window.reopen()
        clear_caps()
    return state


def _rank_callbacks(ctx: RunContext, sources: list, share: float, traced) -> list:
    """The per-rank report and profiled windows of `ctx`'s rank over `sources`."""
    k, built = ctx.k, ctx.built
    return build_rank_callbacks(
        k,
        arch=built.backbone.cfg,
        mode="full" if is_full(k) else "adapter",
        recompute=getattr(built.plan, "checkpoint", "none") != "none",
        rank=ctx.rank,
        world=ctx.world,
        data_world=ctx.data_world,
        device=ctx.device,
        output_dir=ctx.output_dir,
        sources=sources,
        share=share,
        extra_devices=traced,
    )


def _inputs(k: dict, built, device, loader, writer):
    """(loader, batcher): the encode-ahead wrappers (latents, then text) around the
    training loader and the batcher that moves its batches to `device`."""
    dataset = loader.dataset
    attach_text_prewarm(k, built, dataset)
    latent_ahead = build_latent_ahead(k, built, loader)
    loader = loader if latent_ahead is None else latent_ahead
    ahead = build_ahead(k, built, loader)
    batcher = build_batcher(k, built, device, writer, dataset, ahead, latent_ahead)
    return (loader if ahead is None else ahead), batcher


def _validation_batches(k: dict, built, device, kd, tokens, rank, world) -> list:
    """The fixed validation batches on `device` (none when VALIDATE_EVERY is 0; the
    encode feed's under the feed)."""
    if k["VALIDATE_EVERY"] <= 0:
        return []
    if built.feed is not None:
        count = int(k["VALIDATION"].get("batches", 4))
        return fixed_batches(
            built.feed.validation(), build_batcher(k, built, device), count
        )
    val_loader = writer_first(
        rank,
        world,
        partial(build_loader, validation_data_config(kd), rank, world, tokens),
    )
    return fixed_batches(
        val_loader,
        build_batcher(k, built, device, dataset=val_loader.dataset),
        int(k["VALIDATION"].get("batches", 4)),
    )


def _parts(ctx: RunContext, live: tuple, names) -> Parts:
    """The checkpoint parts `ctx.layout` builds over `live` = (ema, bundle, step,
    loader)."""

    def extra() -> dict:
        return {"world": ctx.world}

    return ctx.layout.parts(ctx, (ctx.built.trainable, *live), names, extra)


def export_into(
    k: dict, layout: Layout, built, parts, step: int, root: Path, is_main: bool
) -> None:
    """Every export file of the trained state at `step` under `root` (rank 0 writes;
    every rank takes part in the stage merge and the FSDP2 unshard)."""
    parts.before_export()
    with layout.full_weights(built.backbone):
        if is_main:
            meta = {"kohakumoss_step": str(step), "kohakumoss_algo": str(k["ALGO"])}
            for path in export(k, built, root, meta):
                log.info(f"exported {path}")


def _export(
    k: dict, layout: Layout, built, parts, state, output_dir: str, is_main: bool
) -> None:
    """The run's final exports under `<output>/export/` (none with SAVE_FORMAT
    'none')."""
    if k["SAVE_FORMAT"] == "none":
        return
    root = Path(output_dir) / "export"
    export_into(k, layout, built, parts, state.step, root, is_main)


def checkpoint_exports(k: dict, layout: Layout, built, parts, is_main: bool):
    """`after_save(step, path)` of the CheckpointManager: the exports into
    `<checkpoint>/export/` (CHECKPOINT_EXPORT; every rank calls it, rank 0 writes)."""
    if not k["CHECKPOINT_EXPORT"] or k["SAVE_FORMAT"] == "none":
        return nothing

    def after_save(step: int, path: Path | None) -> None:
        root = (path or Path()) / "export"
        export_into(k, layout, built, parts, step, root, is_main)

    return after_save


def close_quietly(writer) -> None:
    """End of a run that raised: flush and stop the online cache writer, logging (not
    raising) its own failure so the run's error is the one reported."""
    if writer is None:
        return
    try:
        writer.close()
    except Exception as err:  # noqa: BLE001 - the run's own exception is propagating
        log.warning(f"online cache writer did not close cleanly: {err}")
