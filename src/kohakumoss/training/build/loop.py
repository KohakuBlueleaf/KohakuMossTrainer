"""Loop-side build: objective + step, optimiser bundle, EMA, callbacks, driver."""

from kohakumoss.diffusion.objective import LOSSES, WEIGHTINGS, TokenNormalizer
from kohakumoss.diffusion.timestep import TIMESTEPS
from kohakumoss.registry import build, import_object
from kohakumoss.training.build import integration as I
from kohakumoss.training.build.kernels import retune_callbacks
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
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers import DRIVERS, Parts
from kohakumoss.training.drivers.pipeline import PipelineDriver, ThreadedDriver
from kohakumoss.training.ema import build_ema, ema_is_power
from kohakumoss.training.optim import build_bundle
from kohakumoss.training.step import (
    DeviceBatcher,
    OnlineLatents,
    OnlineText,
    TokenMaskHook,
    TrainStep,
    token_counter,
)


def is_masked(k: dict) -> bool:
    """The loss reads a per-token mask (only `masked_mse` does)."""
    return k["LOSS"] == "masked_mse"


def build_step(
    k: dict, built, normalizer: TokenNormalizer, autocast, device, sampler=None
):
    """TrainStep with the loss, weighting, sampler and mask handling bound."""
    masked = is_masked(k)
    loss = build(
        {"name": k["LOSS"], "masked": masked, **k["LOSS_KWARGS"]},
        LOSSES,
        token_mse=built.backbone.ops.token_mse,
    )
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
        pre_forward=tlora_hook(k),
    )


TLORA_MASK_KEYS = ("min_rank", "alpha", "max_timestep")


def tlora_hook(k: dict):
    """T-LoRA's per-step rank mask (None for every other algorithm)."""
    if k["ALGO"] != "tlora":
        return None
    kw = {key: k["ALGO_KWARGS"][key] for key in TLORA_MASK_KEYS if key in k["ALGO_KWARGS"]}
    return TokenMaskHook(import_object(I.TLORA_MASK), int(k["DIM"]), **kw)


def build_batcher(k: dict, built, device, writer=None) -> DeviceBatcher:
    """Cached tensors from the loader, or online encoding with the built encoders when
    a cache is not 'offline'; `writer` (a CacheWriter) receives every fresh encoding."""
    kw = {"masked": is_masked(k)}
    if k["TEXT_CACHE"] != "offline":
        kw["text"] = OnlineText(built.text_encoder.encode)
    if k["LATENT_CACHE"] != "offline":
        kw["latents"] = OnlineLatents(built.vae.encode_images)
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
    return import_object(I.DATA_CACHE_WRITER)(loader.dataset)


def _chain(*fns):
    fns = tuple(f for f in fns if f is not None)

    def run() -> None:
        for f in fns:
            f()

    return run


ADAMW_BY_STATE = {
    "gpu": "adamw",
    "8bit": "adamw8bit",
    "cpu": "cpu_adamw",
    "stream": "host_state_adamw",
}


def _optimizer_name(k: dict, built) -> str:
    """OPTIMIZER, with `adamw` following the plan's optimiser-state placement (host
    masters of a trainable weight stream force the host AdamW)."""
    if k["OPTIMIZER"] != "adamw":
        return k["OPTIMIZER"]
    state = "cpu" if built.engines.host_params() else built.plan.optim_state
    return ADAMW_BY_STATE[state]


def _offload_parts(grads, host_optimizer: bool, fused: bool) -> dict:
    """Bundle hooks for gradient offload: host-optimiser (reads the host sums),
    device optimiser (sums put back on device before the step), or in-backward finish."""
    host = list(grads.host.values())
    match (fused, host_optimizer):
        case (True, _):
            return {"after": grads.zero}
        case (False, True):
            return {"before": grads.flush, "after": grads.zero, "grads": lambda: host}
        case _:
            return {"before": _chain(grads.flush, grads.to_device), "after": grads.zero}


def build_optim(k: dict, built, device, policy, hooks, keep: set[int] | None = None):
    """(bundle, accumulate-context) over the trainable groups (restricted to the
    parameter ids in `keep` for pipeline stages), with the plan's optimiser placement,
    fused per-parameter steps and gradient offload wired."""
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
    optimizer = _optimizer_name(k, built)
    engines = built.engines
    plan = built.plan
    fused = bool(plan.fused_step)
    params = [p for g in groups for p in g["params"]]
    host_masters = {id(p) for p in engines.host_params()}
    device_params = [p for p in params if id(p) not in host_masters]
    grads = None
    extra = {}
    if plan.grad_offload:
        grads = import_object(I.MEMORY_GRAD_OFFLOAD)(engines, device_params, plan.pinned)
        extra = _offload_parts(grads, optimizer == "cpu_adamw", fused)
    bundle = build_bundle(
        groups,
        device,
        optimizer,
        hyper,
        k["SCHEDULE"],
        k["STEPS"],
        policy,
        k["GRAD_CLIP"],
        fused_step=fused,
        grad_norm=hooks.grad_norm,
        agree=hooks.agree,
        before=_chain(hooks.before, engines.before_step, extra.get("before")),
        after=_chain(engines.after_step, extra.get("after")),
        reduce_loss=hooks.reduce_loss,
        grads=extra.get("grads"),
    )
    if optimizer == "cpu_adamw" and grads is not None:
        _host_optimizer(bundle).use_host_grads(grads)
    if fused and grads is not None:
        bundle.attach(grads, finishes=True)
    elif fused:
        bundle.attach(import_object(I.MEMORY_FUSED)(engines, params, bundle.step_param, 0.0))
    accumulate = bundle.arm if fused else _offload_arm(grads, hooks.accumulate)
    return bundle, accumulate


def _host_optimizer(bundle):
    return bundle.shared if hasattr(bundle, "shared") else bundle.optimizer


def _offload_arm(grads, accumulate):
    """Offload this micro-batch's gradients (every micro-batch) and keep the parallel
    gradient-sync context."""
    if grads is None:
        return accumulate

    def arm(last: bool):
        grads.arm(last, None)
        return accumulate(last)

    return arm


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
) -> CallbackRunner:
    """Every per-step action of the run, with its cadence, in firing order."""
    every = k["LOG_EVERY"]
    modes = (bundle.eval_mode, bundle.train_mode)
    mode = "full" if k["ALGO"] == "full" else "adapter"
    recompute = getattr(built.plan, "checkpoint", "none") != "none"
    cbs = retune_callbacks(k)
    cbs.append(Throughput(every, built.backbone.cfg, mode, recompute, device))
    if device.type == "cuda":
        cbs.append(Vram(every, device))
    if k["SAMPLE_EVERY"] > 0 and is_main:
        pipe = import_object(I.PIPELINE).from_components(
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
            )
        )
    cbs += [InstabilityDump(output_dir), LogMetrics(every, logger)]
    if k["SAVE_EVERY"] > 0:
        cbs.append(SaveCheckpoint(k["SAVE_EVERY"], checkpoints, parts))
    if k["SAVE_MINUTES"] > 0:
        cbs.append(TimedCheckpoint(k["SAVE_MINUTES"], checkpoints, parts))
    return CallbackRunner(cbs)


def build_checkpoints(k, output_dir, ema, is_main, writer=None) -> CheckpointManager:
    """Checkpoints; the online cache writer is flushed before each save."""
    kw = {} if writer is None else {"before_save": writer.flush}
    return CheckpointManager(
        output_dir,
        k["KEEP_LAST"],
        snapshot_ema=ema_is_power(ema),
        is_main=is_main,
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
    )
    match (k["PARALLEL"], k["DRIVER"]):
        case ("threaded", _):
            return ThreadedDriver(**common, pipe=engine)
        case ("pipeline", _):
            return PipelineDriver(**common, engine=engine, world=world)
        case (_, "native"):
            return DRIVERS.get("native")(**common, accumulate=accumulate)
    strategy = "ddp" if k["PARALLEL"] == "ddp" else "auto"
    return DRIVERS.get("lightning")(
        **common, devices=len(k["DEVICES"]), strategy=strategy
    )


def fixed_batches(loader, batcher, count: int) -> list:
    """The first `count` batches of a loader, moved to device once."""
    out = []
    for hb in loader:
        out.append(batcher(hb))
        if len(out) >= count:
            break
    return out
