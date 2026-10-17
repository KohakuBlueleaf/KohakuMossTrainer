"""Loop-side build: objective + step, optimiser bundle, EMA, callbacks, driver."""

import torch.distributed as dist

from kohakumoss.diffusion.objective import WEIGHTINGS, TokenNormalizer
from kohakumoss.diffusion.timestep import TIMESTEPS
from kohakumoss.inference.pipeline import MossPipeline
from kohakumoss.registry import build
from kohakumoss.training.build.kernels import retune_callbacks
from kohakumoss.training.build.objective import build_loss, is_masked
from kohakumoss.training.build.pipelines import filter_groups
from kohakumoss.training.callbacks import (
    CallbackRunner,
    InstabilityDump,
    LogMetrics,
    SampleGrid,
    SaveCheckpoint,
    Throughput,
    TimedCheckpoint,
    Validation,
    Vram,
)
from kohakumoss.training.callbacks.runner import rank0_decision
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers import DRIVERS, Parts
from kohakumoss.training.drivers.pipeline import PipelineDriver, ThreadedDriver
from kohakumoss.training.ema import build_ema, ema_is_power
from kohakumoss.training.memory.optimizers import build_memory_optim
from kohakumoss.training.step import (
    DeviceBatcher,
    OnlineLatents,
    OnlineText,
    TrainStep,
    token_counter,
)
from kohakumoss_data import CacheWriter, DataConfig


def build_step(
    k: dict,
    built,
    normalizer: TokenNormalizer,
    autocast,
    device,
    sampler=None,
    stream_rank: int = 0,
):
    """TrainStep with the loss, weighting, sampler and mask handling bound; its sigma /
    noise streams are those of `stream_rank`."""
    masked = is_masked(k)
    loss = build_loss(k, built)
    sampler = sampler or build(
        {"name": k["TIMESTEP"], **k["TIMESTEP_KWARGS"]}, TIMESTEPS
    )
    return TrainStep(
        backbone=built.backbone,
        loss_fn=loss,
        weighting=build(k.get("WEIGHTING", "none"), WEIGHTINGS),
        sampler=sampler,
        normalizer=normalizer,
        autocast=autocast,
        device=device,
        seed=k["SEED"],
        masked=masked,
        pre_forward=built.trainable.begin_step,
        rank=stream_rank,
    )


def build_batcher(k: dict, built, device, writer=None, dataset=None) -> DeviceBatcher:
    """Cached tensors from the loader, or online encoding with the built encoders when
    a cache is not 'offline'; `writer` (a CacheWriter) receives every fresh encoding.
    With an `online` cache, fresh encodings are rounded as `dataset`'s cache stores them,
    so a miss trains on what a later hit returns."""
    kw = {"masked": is_masked(k)}
    if k["TEXT_CACHE"] != "offline":
        rounding = (
            {"as_cached": dataset.texts.as_cached}
            if k["TEXT_CACHE"] == "online" and dataset is not None
            else {}
        )
        kw["text"] = OnlineText(built.text_encoder.encode, **rounding)
    if k["LATENT_CACHE"] != "offline":
        rounding = (
            {"as_cached": dataset.latents.as_cached}
            if k["LATENT_CACHE"] == "online" and dataset is not None
            else {}
        )
        kw["latents"] = OnlineLatents(built.vae.encode_images, **rounding)
    if writer is not None:
        kw["on_encoded"] = writer
    return DeviceBatcher(device, built.backbone.cfg.rope_axes, **kw)


def build_writer(k: dict, loader, writes: bool):
    """The online cache's background writer over the loader's dataset; None when no
    cache is online or this rank does not write (pipeline ranks > 0 draw rank 0's
    batches)."""
    online = "online" in (k["LATENT_CACHE"], k["TEXT_CACHE"])
    if not (online and writes):
        return None
    return CacheWriter(loader.dataset)


def build_optim(k: dict, built, device, policy, hooks, keep: set[int] | None = None):
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
        policy,
        k["GRAD_CLIP"],
        hooks,
    )


def build_ema_for(k: dict, built, device, names: set[str] | None = None):
    """EMA over the trainable set (only `names` when given: pipeline stages)."""
    named = list(built.trainable.named_trainable())
    if names is not None:
        named = [(n, p) for n, p in named if n in names]
    return build_ema(named, k["EMA"], device=device)


def sample_settings(k: dict) -> dict:
    """MossPipeline.generate kwargs from the SAMPLE_* knobs (one seed per prompt)."""
    prompts = list(k["SAMPLE_PROMPTS"])
    seeds = list(k["SAMPLE_SEEDS"]) or [0]
    return {
        "prompts": prompts,
        "negative_prompts": [k.get("SAMPLE_NEGATIVE", "")] * len(prompts),
        "sizes": tuple(k["SAMPLE_SIZE"]),
        "steps": k["SAMPLE_STEPS"],
        "cfg": k["SAMPLE_CFG"],
        "shift": k["SAMPLE_SHIFT"],
        "skip_layers": list(k["SAMPLE_SKIP_LAYERS"]),
        "seeds": [seeds[i % len(seeds)] for i in range(len(prompts))],
    }


class PipelineImages:
    """`generate(**settings).images` of a MossPipeline over the live training model."""

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, **settings) -> list:
        return self.pipe.generate(**settings).images


def build_callbacks(
    k,
    built,
    bundle,
    logger,
    output_dir,
    device,
    is_main,
    make_step,
    val_batches,
    checkpoints,
    parts,
    reduce_loss=None,
) -> CallbackRunner:
    """Every per-step action of the run, with its cadence, in firing order;
    `reduce_loss` is the parallel layout's loss reduction (validation uses it)."""
    every = k["LOG_EVERY"]
    modes = (bundle.eval_mode, bundle.train_mode)
    mode = "full" if k["ALGO"] == "full" else "adapter"
    recompute = getattr(built.plan, "checkpoint", "none") != "none"
    cbs = retune_callbacks(k)
    cbs.append(Throughput(every, built.backbone.cfg, mode, recompute, device))
    if device.type in ("cuda", "xpu"):
        cbs.append(Vram(every, device))
    if k["SAMPLE_EVERY"] > 0 and is_main:
        pipe = MossPipeline.from_components(
            built.backbone, built.text_encoder, built.vae
        )
        cbs.append(
            SampleGrid(
                k["SAMPLE_EVERY"],
                PipelineImages(pipe),
                sample_settings(k),
                output_dir,
                logger,
                built.backbone,
                modes,
            )
        )
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
    cbs += [InstabilityDump(output_dir), LogMetrics(every, logger)]
    if k["SAVE_EVERY"] > 0:
        cbs.append(SaveCheckpoint(k["SAVE_EVERY"], checkpoints, parts))
    if k["SAVE_MINUTES"] > 0:
        agree = (
            {"agree": rank0_decision}
            if dist.is_available() and dist.is_initialized()
            else {}
        )
        cbs.append(
            TimedCheckpoint(
                k["SAVE_MINUTES"], checkpoints, parts, every=k["LOG_EVERY"], **agree
            )
        )
    return CallbackRunner(cbs)


def validation_data_config(k: dict):
    """The validation set's DataConfig: the run's data knobs over VALIDATION['datasets'],
    every sample conditional (no text dropout)."""
    return DataConfig.from_knobs(
        {**k, "DATASETS": k["VALIDATION"]["datasets"], "TEXT_DROP": 0.0}
    )


def build_checkpoints(
    k, output_dir, ema, is_main, writer=None, rank: int = 0, devices=()
) -> CheckpointManager:
    """Checkpoints (RNG kept per rank and per device in `devices`); the online cache
    writer is flushed before each save."""
    kw = {} if writer is None else {"before_save": writer.flush}
    return CheckpointManager(
        output_dir,
        k["KEEP_LAST"],
        snapshot_ema=ema_is_power(ema),
        is_main=is_main,
        rank=rank,
        devices=devices,
        **kw,
    )


def build_driver(
    k,
    batcher: DeviceBatcher,
    normalizer,
    device,
    runner: CallbackRunner,
    ckpt,
    parts: Parts,
    accumulate,
    engine=None,
    world: int = 1,
):
    """Native / Lightning loop, or the pipeline drivers for PARALLEL threaded /
    pipeline (micro-batches per step = MICROBATCHES there)."""
    pipelined = k["PARALLEL"] in ("pipeline", "threaded")
    common = dict(
        steps=k["STEPS"],
        grad_acc=k["MICROBATCHES"] if pipelined else k["GRAD_ACC"],
        batcher=batcher,
        normalizer=normalizer,
        device=device,
        runner=runner,
        checkpoints=ckpt,
        parts=parts,
        count_fn=token_counter(is_masked(k)),
        end_step=parts.trainable.end_step,
    )
    match (k["PARALLEL"], k["DRIVER"]):
        case ("threaded", _):
            return ThreadedDriver(**common, pipe=engine)
        case ("pipeline", _):
            return PipelineDriver(**common, engine=engine, world=world)
        case (_, "native"):
            return DRIVERS.get("native")(**common, accumulate=accumulate)
    return DRIVERS.get("lightning")(**common, accumulate=accumulate)


def fixed_batches(loader, batcher, count: int) -> list:
    """The first `count` batches of a loader, moved to device once."""
    out = []
    for hb in loader:
        out.append(batcher(hb))
        if len(out) >= count:
            break
    return out
