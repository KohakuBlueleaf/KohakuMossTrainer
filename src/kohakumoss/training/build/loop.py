"""Loop-side build: objective + step, optimiser bundle, EMA, callbacks, driver."""

import contextlib
from collections.abc import Callable
from functools import partial
from pathlib import Path

from kohakumoss.diffusion.objective import WEIGHTINGS, TokenNormalizer
from kohakumoss.diffusion.timestep import TIMESTEPS
from kohakumoss.inference.pipeline import MossPipeline
from kohakumoss.logging.log import get_logger, interactive, progress_enabled
from kohakumoss.memory.device_use import NoDeviceUse
from kohakumoss.memory.release import release_callbacks
from kohakumoss.memory.timeline import NoTimeline
from kohakumoss.parallel.launch import dist_rank, ranks_meet
from kohakumoss.training.build.objective import build_loss, is_masked
from kohakumoss.training.build.pipelines import filter_groups
from kohakumoss.training.callbacks import (
    CallbackRunner,
    FeedWait,
    InstabilityDump,
    LogMetrics,
    SampleGrid,
    SaveCheckpoint,
    StepLine,
    TextEncodeStats,
    Throughput,
    TimedCheckpoint,
    VaeEncodeStats,
    Validation,
    Vram,
    VramAtStepOne,
)
from kohakumoss.training.callbacks.finite import NonFiniteGuard
from kohakumoss.training.callbacks.kernels import (
    DecisionsLive,
    retune_callbacks,
    speed_callbacks,
    vram_callbacks,
)
from kohakumoss.training.callbacks.progress import ProgressBar
from kohakumoss.training.callbacks.repro import StepRecorder
from kohakumoss.training.callbacks.runner import MeetRanks, rank0_decision
from kohakumoss.training.callbacks.startup import StepOneBreakdown
from kohakumoss.training.callbacks.transfer import TransferStats, transfer_sources
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.context import RunContext
from kohakumoss.training.drivers import Parts
from kohakumoss.training.ema import build_ema, ema_is_power
from kohakumoss.training.ema.averagers import BackgroundEMA, NoEMA, host_averaged
from kohakumoss.training.knobs import is_full
from kohakumoss.training.memory.optimizers import build_memory_optim
from kohakumoss.training.step import (
    DeviceBatcher,
    OnlineLatents,
    OnlineText,
    TrainStep,
    token_counter,
)
from kohakumoss.training.textenc import TextAhead
from kohakumoss.training.vae import LatentAhead
from kohakumoss_common.calls import nothing
from kohakumoss_common.registry import build
from kohakumoss_data import CacheWriter, DataConfig
from kohakumoss_kernels.dist import initialized
from kohakumoss_kernels.tuning.window import tuning_window

log = get_logger(__name__)


def build_step(
    k: dict,
    built,
    normalizer: TokenNormalizer,
    autocast,
    device,
    sampler=None,
    data_rank: int = 0,
    recorder=None,
    forward=None,
):
    """TrainStep with the loss, weighting, sampler and mask handling bound; its sigma /
    noise streams are those of `data_rank`; `recorder` (a StepRecorder) keeps each
    prepared micro-batch; `forward` is the module the step calls instead of the
    backbone (the backbone under torch DDP)."""
    masked = is_masked(k)
    loss = build_loss(k, built)
    sampler = sampler or build(
        {"name": k["TIMESTEP"], **k["TIMESTEP_KWARGS"]}, TIMESTEPS
    )
    return TrainStep(
        backbone=built.backbone if forward is None else forward,
        loss_fn=loss,
        weighting=build(k.get("WEIGHTING", "none"), WEIGHTINGS),
        sampler=sampler,
        normalizer=normalizer,
        autocast=autocast,
        device=device,
        seed=k["SEED"],
        masked=masked,
        pre_forward=built.trainable.begin_step,
        rank=data_rank,
        keep=None if recorder is None else recorder.keep,
    )


def build_recorder(k: dict, built):
    """The non-finite step recorder when NONFINITE_PATIENCE stops runs, else None (full
    finetune: it writes only the parameters the step reports non-finite)."""
    if k["NONFINITE_PATIENCE"] <= 0:
        return None
    return StepRecorder(built.trainable, only_named=is_full(k))


def build_batcher(
    k: dict, built, device, writer=None, dataset=None, ahead=None, latent_ahead=None
) -> DeviceBatcher:
    """Cached tensors from the loader, or online encoding with the built encoders when
    a cache is not 'offline' (text / latents through the encode-ahead passes `ahead` /
    `latent_ahead` when given);
    `writer` (a CacheWriter) receives every fresh encoding. With an `online` cache,
    fresh encodings are rounded as `dataset`'s cache stores them, so a miss trains on
    what a later hit returns. Under the encode feed every batch arrives encoded (the
    feed rounds and writes), so the batcher only moves it."""
    kw = {"masked": is_masked(k)}
    if built.feed is not None:
        return DeviceBatcher(device, built.backbone.cfg.rope_axes, **kw)
    if k["TEXT_CACHE"] != "offline":
        rounding = (
            {"as_cached": dataset.texts.as_cached}
            if k["TEXT_CACHE"] == "online" and dataset is not None
            else {}
        )
        source = None if ahead is None else ahead.take
        kw["text"] = OnlineText(built.text_encoder.encode, source=source, **rounding)
    if k["LATENT_CACHE"] != "offline":
        rounding = (
            {"as_cached": dataset.latents.as_cached}
            if k["LATENT_CACHE"] == "online" and dataset is not None
            else {}
        )
        source = None if latent_ahead is None else latent_ahead.take
        kw["latents"] = OnlineLatents(
            built.vae.encode_images, source=source, **rounding
        )
    if writer is not None:
        kw["on_encoded"] = writer
    return DeviceBatcher(device, built.backbone.cfg.rope_axes, **kw)


def build_ahead(k: dict, built, loader):
    """The encode-ahead loader wrapper when text is encoded online (one text-encoder pass
    per TEXT_ENCODE['ahead'] host batches), else None."""
    if k["TEXT_CACHE"] == "offline" or built.text_encoder is None:
        return None
    plan = getattr(built.text_encoder, "plan", None)
    n = 1 if plan is None else plan.ahead
    on_host = plan is not None and plan.memory == "cpu"
    report = (
        partial(host_pass_line, "text encoder", "prompts", n) if on_host else nothing
    )
    return TextAhead(loader, built.text_encoder.encode, n, report)


def build_latent_ahead(k: dict, built, loader):
    """The latent encode-ahead wrapper when latents are encoded online by a placed VAE
    (one VAE pass per VAE_ENCODE['ahead'] host batches), else None."""
    if k["LATENT_CACHE"] == "offline" or built.feed is not None:
        return None
    if not hasattr(built.vae, "encode_batches"):
        return None
    n = built.vae.plan.ahead
    on_host = built.vae.plan.memory == "cpu"
    report = partial(host_pass_line, "VAE", "images", n) if on_host else nothing
    return LatentAhead(loader, built.vae, n, report)


def host_pass_line(what: str, unit: str, n: int, seconds: float, count: int) -> None:
    """The one INFO line of an encoder that encodes online on the host: what its first
    encode-ahead pass cost."""
    log.info(
        f"{what} on the host encodes online: the first pass ({count} {unit}, {n} host "
        f"batches) took {seconds:.1f} s, about {seconds / n:.1f} s per step"
    )


def build_writer(k: dict, loader, writes: bool, rank: int = 0):
    """The online cache's background writer over the loader's dataset, writing this
    rank's shards; None when no cache is online or this rank does not write (pipeline
    ranks > 0 draw rank 0's batches)."""
    online = "online" in (k["LATENT_CACHE"], k["TEXT_CACHE"])
    if not (online and writes):
        return None
    return CacheWriter(loader.dataset, rank=rank)


def build_optim(k: dict, built, device, prec, hooks, keep: set[int] | None = None):
    """(bundle, accumulate-context) over the trainable groups (restricted to the
    parameter ids in `keep` for pipeline stages) under the memory plan's wiring."""
    hyper = {
        "lr": k["LR"],
        "betas": tuple(k["BETAS"]),
        "eps": k["EPS"],
        **k.get("OPTIMIZER_KWARGS", {}),
    }
    groups = built.trainable.param_groups(
        k["LR"],
        weight_decay=k["WEIGHT_DECAY"],
        rules=k["WD_RULES"],
        use_muon=k["OPTIMIZER"] == "muon",
    )
    if keep is not None:
        groups = filter_groups(groups, keep)
    return build_memory_optim(
        groups,
        built.plan,
        built.engines,
        device,
        k["OPTIMIZER"],
        hyper,
        k["SCHEDULE"],
        k["STEPS"],
        prec,
        k["GRAD_CLIP"],
        hooks,
    )


def build_ema_for(k: dict, built, bundle, names: set[str] | None = None, rank: int = 0):
    """EMA over the trainable set (only `names` when given: pipeline stages), each
    average on its parameter's device (host masters: host); its update first waits for
    the bundle's in-backward block write-back (`ema_before_read`). Under DDP only rank
    0 keeps one; host averages update in the background (`BackgroundEMA`)."""
    if k.get("PARALLEL", "none") == "ddp" and rank != 0:
        return NoEMA()
    named = list(built.trainable.named_trainable())
    if names is not None:
        named = [(n, p) for n, p in named if n in names]
    ema = build_ema(named, k["EMA"])
    if host_averaged(ema):
        ema = BackgroundEMA(ema)
    ema.before_read = ema_before_read(bundle)
    return ema


def ema_before_read(bundle) -> Callable[[], None]:
    """What the EMA waits for before reading the masters: the per-block optimiser's
    write-back to the host masters (`StreamedAdamW.synchronize`), else nothing."""
    blocks = getattr(bundle, "blocks", None)
    return nothing if blocks is None else blocks.synchronize


def sample_settings(k: dict, embeds: list | None = None) -> dict:
    """MossPipeline.generate kwargs from the SAMPLE_* knobs (one seed per prompt); with
    `embeds` (the build-time encodings of the prompts, then the negative) the pipeline
    needs no text encoder."""
    prompts = list(k["SAMPLE_PROMPTS"])
    seeds = list(k["SAMPLE_SEEDS"]) or [0]
    text = {
        "prompts": prompts,
        "negative_prompts": [k.get("SAMPLE_NEGATIVE", "")] * len(prompts),
    }
    if embeds is not None:
        text.update(
            prompt_embeds=embeds[: len(prompts)],
            negative_embeds=[embeds[-1]] * len(prompts),
        )
    return {
        **text,
        "sizes": tuple(k["SAMPLE_SIZE"]),
        "steps": k["SAMPLE_STEPS"],
        "cfg": k["SAMPLE_CFG"],
        "shift": k["SAMPLE_SHIFT"],
        "skip_layers": list(k["SAMPLE_SKIP_LAYERS"]),
        "seeds": [seeds[i % len(seeds)] for i in range(len(prompts))],
    }


class PipelineImages:
    """`generate(**settings).images` of a MossPipeline over the live training model,
    under the run's `autocast` (the training compute dtype). The first call tunes its
    missing kernel configs up to the end of its first sampler step
    (`window.tuning_window`); later calls time nothing."""

    def __init__(self, pipe, autocast=contextlib.nullcontext):
        self.pipe = pipe
        self.autocast = autocast
        self._run = PipelineImages._first

    def __call__(self, **settings) -> list:
        return self._run(self, **settings)

    def _first(self, **settings) -> list:
        self._run = PipelineImages._later
        with tuning_window() as close:
            user = settings.get("step_callback")
            settings["step_callback"] = partial(_closing, close, user)
            return self._later(**settings)

    def _later(self, **settings) -> list:
        with self.autocast():
            return self.pipe.generate(**settings).images


def _closing(close, user, i: int, x) -> None:
    """Sampler step `i` done: end the tuning window, then `user`."""
    close()
    if user is not None:
        user(i, x)


def metrics_every(k: dict) -> int:
    """The metric-row cadence: METRICS_EVERY, or LOG_EVERY when None."""
    rows = k.get("METRICS_EVERY")
    return k["LOG_EVERY"] if rows is None else int(rows)


def bar_shown(k: dict) -> bool:
    """Whether the run shows the progress bar (PROGRESS, a TTY or FORCE_COLOR, the
    level at INFO or lower); without it the console gets StepLine."""
    return bool(k.get("PROGRESS", True)) and interactive() and progress_enabled()


def build_callbacks(
    ctx: RunContext,
    bundle,
    logger,
    make_step,
    val_batches,
    checkpoints,
    parts,
    *,
    reduce_loss=None,
    autocast=contextlib.nullcontext,
    timeline=NoTimeline(),
    device_use=NoDeviceUse(),
    rank_callbacks=(),
    recorder=None,
) -> CallbackRunner:
    """Every per-step action of the run (`ctx`: knobs, model, device, rank, run
    directory), with its cadence, in firing order;
    `reduce_loss` is the parallel layout's loss reduction (validation uses it),
    `autocast` the run's compute-dtype context (sampling runs under it), `timeline`
    the VRAM timeline the text encoder marks, `device_use` the process's device
    accounting (both null objects on the host), `rank_callbacks` the per-rank report and
    profiled window, before the metric rows, and `recorder` the StepRecorder the
    non-finite guard writes the failing step with."""
    k, built, device = ctx.k, ctx.built, ctx.device
    output_dir, is_main = ctx.output_dir, ctx.is_main
    every = k["LOG_EVERY"]
    rows = metrics_every(k)
    modes = (bundle.eval_mode, bundle.train_mode)
    mode = "full" if is_full(k) else "adapter"
    recompute = getattr(built.plan, "checkpoint", "none") != "none"
    cbs = retune_callbacks(k) + [DecisionsLive()] + speed_callbacks(k)
    text, vae = encode_stats(built)
    cbs.append(StepOneBreakdown(text, vae, output_dir, device))
    cbs.append(Throughput(rows, built.backbone.cfg, mode, recompute, device))
    sources = transfer_sources(built.engines, bundle)
    if sources:
        cbs.append(TransferStats(rows, sources))
    if hasattr(text, "stats"):
        cbs.append(TextEncodeStats(rows, text, device))
    if hasattr(vae, "stats"):
        cbs.append(VaeEncodeStats(rows, vae))
    if built.feed is not None:
        cbs.append(FeedWait(rows, built.feed))
    cbs += vram_callbacks(k, device)
    cbs += release_callbacks(device, rows, k["STEP_RELEASE"])
    cbs.append(Vram(rows, timeline, device_use))
    cbs.append(VramAtStepOne(device_use))
    if k["SAMPLE_EVERY"] > 0 and is_main:
        pipe = MossPipeline.from_components(
            built.backbone,
            built.text_encoder,
            built.vae,
            memory_plan=built.plan,
            weight_stream=built.engines.weight_stream,
        )
        cbs.append(
            SampleGrid(
                k["SAMPLE_EVERY"],
                PipelineImages(pipe, autocast),
                sample_settings(k, built.sample_embeds),
                output_dir,
                logger,
                built.backbone,
                modes,
            )
        )
    if k["SAMPLE_EVERY"] > 0 and dist_rank()[1] > 1:
        cbs.append(MeetRanks(k["SAMPLE_EVERY"], "sampled", ranks_meet))
    if k["VALIDATE_EVERY"] > 0:
        cbs.append(
            Validation(
                k["VALIDATE_EVERY"],
                make_step,
                val_batches,
                tuple(k["VALIDATION"].get("sigmas", (0.25, 0.5, 0.75))),
                int(k["VALIDATION"].get("seed", 0)),
                built.backbone,
                modes,
                **({} if reduce_loss is None else {"reduce_loss": reduce_loss}),
                end_step=built.trainable.end_step,
            )
        )
    cbs.append(InstabilityDump(output_dir, bundle=bundle))
    shown = bar_shown(k)
    if is_main and not shown:
        cbs.append(StepLine(every))
    cbs += list(rank_callbacks)
    cbs.append(LogMetrics(rows, logger))
    if k["NONFINITE_PATIENCE"] > 0:
        floor = getattr(bundle.scaler, "floor", bundle.scaler.scale)
        dump = str(Path(output_dir) / "instability")
        record = nothing if recorder is None else recorder.write
        forget = nothing if recorder is None else recorder.drop
        cbs.append(NonFiniteGuard(k["NONFINITE_PATIENCE"], floor, dump, record, forget))
    if is_main and shown:
        cbs.append(ProgressBar(k["STEPS"], device, timeline, device_use))
    if k["SAVE_EVERY"] > 0:
        cbs.append(SaveCheckpoint(k["SAVE_EVERY"], checkpoints, parts))
    if k["SAVE_MINUTES"] > 0:
        agree = {"agree": rank0_decision} if initialized() else {}
        cbs.append(
            TimedCheckpoint(
                k["SAVE_MINUTES"], checkpoints, parts, every=k["LOG_EVERY"], **agree
            )
        )
    return CallbackRunner(cbs, last=k["STEPS"])


def encode_stats(built) -> tuple:
    """(text, VAE) objects whose `stats` the encode callbacks read: the built encoders,
    or the encode feed's statistics under the feed."""
    if built.feed is not None:
        return built.feed.text, built.feed.vae
    return built.text_encoder, built.vae


def validation_data_config(k: dict):
    """The validation set's DataConfig: the run's data knobs over VALIDATION['datasets'],
    every sample conditional."""
    return DataConfig.from_knobs(
        {**k, "DATASETS": k["VALIDATION"]["datasets"], "TEXT_DROP": 0.0}
    )


def final_checkpoint(k: dict) -> bool:
    """Whether the run ends with a resume checkpoint: FINAL_CHECKPOINT, where "auto" is
    on when periodic checkpoints are (SAVE_EVERY or SAVE_MINUTES > 0)."""
    match k.get("FINAL_CHECKPOINT", "auto"):
        case "auto":
            return k["SAVE_EVERY"] > 0 or k["SAVE_MINUTES"] > 0
        case choice:
            return bool(choice)


def build_checkpoints(
    k, output_dir, ema, is_main, writer=None, rank: int = 0, devices=()
) -> CheckpointManager:
    """Checkpoints (RNG kept per rank and per device in `devices`); the online cache
    writer is flushed before each save."""
    kw = {} if writer is None else {"before_save": writer.flush}
    return CheckpointManager(
        output_dir,
        k["KEEP_LAST"],
        final=final_checkpoint(k),
        snapshot_ema=ema_is_power(ema),
        is_main=is_main,
        rank=rank,
        devices=devices,
        **kw,
    )


def resumed_from(k: dict, output_dir: str):
    """The checkpoint RESUME names in `output_dir` (None: a fresh run), before the run's
    checkpoints are built."""
    return CheckpointManager(output_dir, 0, is_main=False).resolve(k["RESUME"])


def build_driver(
    ctx: RunContext,
    batcher: DeviceBatcher,
    normalizer,
    runner: CallbackRunner,
    ckpt,
    parts: Parts,
    accumulate,
    engine=None,
):
    """The driver `ctx.layout` (a training.layouts.Layout) builds: the DRIVER loop, or
    a pipeline driver (micro-batches per step = MICROBATCHES there)."""
    k, layout, device = ctx.k, ctx.layout, ctx.device
    common = dict(
        steps=k["STEPS"],
        grad_acc=k["MICROBATCHES"] if layout.pipelined else k["GRAD_ACC"],
        batcher=batcher,
        normalizer=normalizer,
        device=device,
        runner=runner,
        checkpoints=ckpt,
        parts=parts,
        count_fn=token_counter(is_masked(k)),
        end_step=parts.trainable.end_step,
    )
    return layout.driver(ctx, common, accumulate, engine)


def fixed_batches(loader, batcher, count: int) -> list:
    """The first `count` batches of a loader, moved to device once."""
    out = []
    for hb in loader:
        out.append(batcher(hb))
        if len(out) >= count:
            break
    return out
