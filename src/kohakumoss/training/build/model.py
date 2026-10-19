"""Model-side build: backbone, fold, memory plan, trainable set, encoders, export."""

import contextlib
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch

from kohakumoss.adapters.build.apply import apply as apply_adapters
from kohakumoss.adapters.build.config import config_from_knobs
from kohakumoss.encoders.specs import cache_specs
from kohakumoss.encoders.text import QwenTextEncoder, text_plan
from kohakumoss.encoders.text.encoder import (
    default_auto_memory,
    default_w4a8,
    resolve_text_plan,
)
from kohakumoss.encoders.text.plan import DTYPES as TEXT_DTYPES
from kohakumoss.encoders.text.plan import MEMORY_ALIASES as TEXT_MEMORY_ALIASES
from kohakumoss.encoders.vae import FluxVae, resolve_vae_dtype
from kohakumoss.logging.log import get_logger, hl
from kohakumoss.memory.apply import apply_to_backbone
from kohakumoss.memory.estimate import Workload, estimate
from kohakumoss.memory.planner import NothingFits, auto_plan
from kohakumoss.memory.profiles import ORDER, catalog_name
from kohakumoss.memory.profiles import resolve as resolve_profile
from kohakumoss.model import load_backbone, resolve_config
from kohakumoss.model.numerics.fp16fix import apply_fp16_fix
from kohakumoss.model.opset import OpSet
from kohakumoss.offload.accel import module as accel_module
from kohakumoss.offload.accel import streamed, with_index
from kohakumoss.offload.encoders.report import encoder_lines
from kohakumoss.offload.encoders.vae import MEMORY_ALIASES as VAE_MEMORY_ALIASES
from kohakumoss.offload.encoders.vae import place_vae
from kohakumoss.parallel.launch import dist_rank, gate_name, rank0_value
from kohakumoss.precision import apply_param_dtypes
from kohakumoss.training.build.text_estimate import pass_tokens, planned_text_sizes
from kohakumoss.training.ema.averagers import decay_count
from kohakumoss.training.exports import export_files, fused_qkv, write_exports
from kohakumoss.training.full import FullTrainable
from kohakumoss.training.knobs import (
    is_full,
    memory_overrides,
    plan_problems,
    run_name,
)
from kohakumoss.training.optim.groups import last_block_qk_norm
from kohakumoss.training.optim.optimizers.factory import STATE_BYTES
from kohakumoss.training.weights import component, stored_dit_precision
from kohakumoss.types.plans import PrecisionConfig
from kohakumoss.weights import DIT, TEXT, VAE
from kohakumoss_kernels.hardware.current import rng_kept

log = get_logger(__name__)


@dataclass
class Built:
    backbone: torch.nn.Module
    trainable: object
    plan: object
    engines: object
    vae: object | None
    text_encoder: object | None
    sample_embeds: list | None = None


def build_backbone(k: dict, device: torch.device, prec: PrecisionConfig):
    """The backbone loaded from WEIGHTS (TRANSFORMER: the ComfyUI folder's file)."""
    cfg = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    model, _ = load_backbone(
        component(k, DIT),
        device=device,
        dtype=prec.compute,
        keep_fp32=prec.keep_fp32,
        cfg=cfg,
        kernels=k["KERNELS"],
    )
    return model


def workload(
    k: dict, compute: torch.dtype, world: int = 1, text_tokens: int | None = None
):
    """The `auto` planner's per-rank micro-step: MOSS arch of VARIANT, tokens from the
    batching knobs; text tokens per sample: `text_tokens` (the dataset's longest
    caption) when known, else MAX_TEXT_TOKENS."""
    cfg = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    per_txt = max(1, int(text_tokens or k["MAX_TEXT_TOKENS"]))
    per_img = max(1, k["TARGET_AREA"] // (16 * 16))
    if k["BATCHING"] == "count":
        samples = k["BATCH_SIZE"]
        img, txt = samples * per_img, samples * per_txt
    else:
        share = per_img / (per_img + per_txt)
        img = int(k["MAX_TOKENS"] * share)
        txt = k["MAX_TOKENS"] - img
    compute_bytes = torch.empty((), dtype=compute).element_size()
    return Workload(
        arch=cfg,
        img_tokens=img,
        txt_tokens=txt,
        full_ft=is_full(k),
        compute_bytes=compute_bytes,
        world=world,
        sharded=k["PARALLEL"] == "fsdp",
        sharded_grads=fsdp_grads_sharded(k),
        gathered=k["FSDP_RESHARD"] == "backward",
        ema_decays=decay_count(k["EMA"]),
        optim_state_bytes=STATE_BYTES.get(k["OPTIMIZER"], STATE_BYTES["adamw"]),
        master_bytes=compute_bytes if k["OPTIMIZER"] == "adamw16" else 4,
        stored_precision=stored_dit_precision(k),
    )


def fsdp_grads_sharded(k: dict) -> bool:
    """Whether FSDP2 holds only sharded gradients through the step: False when it
    accumulates unsharded ones over GRAD_ACC > 1 micro-batches (FSDP_REDUCE='step')."""
    return not (k["GRAD_ACC"] > 1 and k["FSDP_REDUCE"] == "step")


def auto_text(k: dict, device: torch.device):
    """The `auto` planner's text-encoder term: (TextSizes from its config, tokens per
    pass / call, the placement rule per candidate plan), or None when text is not
    encoded online on `device`."""
    path = _component_path(k, TEXT)
    if k["TEXT_CACHE"] == "offline" or not path or not Path(path).exists():
        return None
    plan, te_device = training_text_plan(k, device)
    if plan.memory == "cpu" or te_device != torch.device(device):
        return None
    sizes = planned_text_sizes(path, plan, te_device)
    return sizes, pass_tokens(plan), partial(encoder_memory, k, "text", "auto")


def auto_profile(
    k: dict, device: torch.device, compute, world: int, text_tokens: int | None
):
    """(MemoryPlan, planner rows) of MEMORY_PROFILE='auto' on this process's `device`
    (SystemExit when nothing fits and MEMORY_AUTO_FORCE is off)."""
    try:
        plan, report = auto_plan(
            workload(k, compute, world, text_tokens),
            device,
            memory_overrides(k),
            trainable_base=is_full(k),
            reject=partial(plan_problems, k),
            text=auto_text(k, device),
            force=bool(k["MEMORY_AUTO_FORCE"]),
        )
    except NothingFits as err:
        raise SystemExit(str(err)) from err
    if report.forced:
        log.warning(
            f"MEMORY_PROFILE='auto' -> {report.chosen}: no profile's estimate fits "
            "the free memory; MEMORY_AUTO_FORCE runs it anyway"
        )
    else:
        log.info(f"MEMORY_PROFILE='auto' -> {hl(report.chosen)}")
    return plan, report.rows


def resolve_memory(
    k: dict,
    device: torch.device,
    compute=torch.bfloat16,
    world=1,
    text_tokens: int | None = None,
):
    """(MemoryPlan, `auto` planner rows or []) for the knobs on `device`;
    `text_tokens` is the dataset's longest caption (`workload`); `auto` counts the
    online text encoder on `device` (`auto_text`) and is planned on rank 0 only: every
    rank of a process group resolves rank 0's profile (`rank0_value`)."""
    trainable_base = is_full(k)
    overrides = memory_overrides(k)
    if k["MEMORY_PROFILE"] == "auto":
        found = {}

        def choose() -> str:
            found["plan"], found["rows"] = auto_profile(
                k, device, compute, world, text_tokens
            )
            return found["plan"].profile

        rank, ranks = dist_rank()
        name = rank0_value(gate_name("auto-plan"), choose, rank, ranks)
        if "plan" in found:
            return found["plan"], found["rows"]
        log.info(f"MEMORY_PROFILE='auto' -> {hl(name)} (rank 0's plan)")
        plan = resolve_profile(
            name, overrides, device=device, trainable_base=trainable_base
        )
        return plan, []
    if k["MEMORY_PROFILE"] in ORDER:
        entry = catalog_name(k["MEMORY_PROFILE"])
        log.info(f"MEMORY_PROFILE={k['MEMORY_PROFILE']!r} is catalog entry {hl(entry)}")
    plan = resolve_profile(
        k["MEMORY_PROFILE"], overrides, device=device, trainable_base=trainable_base
    )
    return plan, []


def _ids(trainable) -> set[int]:
    return {id(p) for p in trainable.trainable_params()}


@contextlib.contextmanager
def seeded_rng(seed: int, device: torch.device):
    """Draws inside start from `seed` on the CPU and on `device`; the caller's
    generators are restored after."""
    with rng_kept([device]):
        torch.random.default_generator.manual_seed(seed)
        if streamed(device):
            index = with_index(device).index
            accel_module(device).default_generators[index].manual_seed(seed)
        yield


def build_trainable(k: dict, backbone, prec: PrecisionConfig, device):
    """FullTrainable for full FT; otherwise the LyCORIS AdapterSet, its init drawn
    from SEED (independent of every draw before it)."""
    if is_full(k):
        frozen = () if k["TRAIN_TEXT_REFINER"] else (r"^text_embedder\.",)
        return FullTrainable(backbone, frozen, last_block_qk_norm(backbone.cfg.depth))
    backbone.requires_grad_(False)
    with seeded_rng(k["SEED"], device):
        return apply_adapters(
            backbone, config_from_knobs(k), dtype=prec.master, device=device
        )


def build_model(
    k: dict,
    device: torch.device,
    prec: PrecisionConfig,
    world: int = 1,
    text_tokens: int | None = None,
) -> Built:
    """Build order of types/context.py up to (not including) the parallel wrap and compile:
    load on the host -> FP16_FIX -> memory plan (quantize, place or stream onto
    `device`, offload, runner) -> trainable -> parameter dtypes -> bind_ops -> encoders.
    `text_tokens`: the dataset's longest caption, for the `auto` plan.
    """
    backbone = build_backbone(k, torch.device("cpu"), prec)
    apply_fp16_fix(backbone, prec.fp16_fix, **prec.fp16_fix_kwargs)
    plan, auto_rows = resolve_memory(k, device, prec.compute, world, text_tokens)
    full = is_full(k)
    engines = apply_to_backbone(
        backbone,
        plan,
        device,
        prec.compute,
        training=True,
        trainable_base=full,
        keep_fp32=prec.keep_fp32,
    )
    if auto_rows:
        engines.reports["auto"] = auto_rows
    trainable = build_trainable(k, backbone, prec, device)
    apply_param_dtypes(backbone, prec, _ids(trainable), k["OPTIMIZER"] == "adamw16")
    engines.exclude_params(trainable.trainable_params())
    backbone.bind_ops(OpSet.resolve(device, prec.compute, k["KERNELS"]))
    vae, text, embeds = build_encoders(k, device, prec, step_reserve(k, plan), plan)
    return Built(backbone, trainable, plan, engines, vae, text, embeds)


def _component_path(k: dict, which: str) -> str:
    """The path of component `which` ("" when a ComfyUI folder holds none: the knob
    checks report that)."""
    try:
        return str(component(k, which))
    except FileNotFoundError:
        return ""


def placement_lines(k: dict, built: Built, device) -> list[tuple[str, dict]]:
    """`(message, fields)` of where the text encoder and the VAE were placed (the
    inference placement lines), or why the run holds none."""
    te, vae = encoder_lines(
        built.text_encoder,
        built.vae,
        _component_path(k, TEXT),
        _component_path(k, VAE),
        device,
    )
    if te is None:
        sampled = " (the sample prompts were encoded at build, then it was freed)"
        why = sampled if k["SAMPLE_EVERY"] > 0 else ""
        te = (f"text encoder: not kept, TEXT_CACHE='offline'{why}", {})
    if vae is None:
        vae = ("VAE: not loaded, LATENT_CACHE='offline' and no sampling", {})
    return [te, vae]


def step_reserve(k: dict, plan) -> int:
    """Bytes the DiT step adds on top of the resident weights (memory estimate at
    MAX_TOKENS): what an `auto` text-encoder placement must leave free."""
    cfg = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    work = Workload(
        cfg,
        img_tokens=int(k["MAX_TOKENS"]),
        txt_tokens=0,
        full_ft=is_full(k),
        stored_precision=stored_dit_precision(k),
    )
    est = estimate(work, plan)
    return max(est.total - est.parts.get("weights", 0), 0)


STREAMED_BASE_ENCODERS = {"vae": "offload"}
ENCODER_KNOBS = {"text": "TEXT_ENCODE", "vae": "VAE_ENCODE"}
ENCODER_ALIASES = {"text": TEXT_MEMORY_ALIASES, "vae": VAE_MEMORY_ALIASES}


def streamed_base(plan) -> bool:
    """The resolved plan keeps the DiT blocks in host memory (streamed)."""
    return plan is not None and plan.stream != "off"


def encoder_memory(k: dict, which: str, default: str, plan=None) -> str:
    """The placement of encoder `which` ("text" / "vae"): its TEXT_ENCODE /
    VAE_ENCODE `memory` when set; else `default` (the text encoder's `auto`: the
    fastest placement that fits next to the DiT step, whatever the plan), and for the
    VAE offload when `plan` streams the DiT blocks. A given `memory` comes in its
    primary spelling (the encoder's `MEMORY_ALIASES`)."""
    given = dict(k.get(ENCODER_KNOBS[which]) or {}).get("memory")
    if given is not None:
        return ENCODER_ALIASES[which].get(given, given)
    if streamed_base(plan):
        return STREAMED_BASE_ENCODERS.get(which, default)
    return default


def encoder_rule(k: dict, which: str, plan) -> str:
    """Where encoder `which`'s placement came from, for the build log."""
    knob = ENCODER_KNOBS[which]
    if "memory" in dict(k.get(knob) or {}):
        return f"{knob}['memory']"
    if which not in STREAMED_BASE_ENCODERS:
        return f"{knob}['memory'] unset: the fastest placement that fits"
    if streamed_base(plan):
        return f"{knob}['memory'] unset, DiT blocks streamed (stream={plan.stream!r})"
    return f"{knob}['memory'] unset, DiT resident"


def training_text_plan(k: dict, device: torch.device, plan=None):
    """(TextEncoderPlan, text-encoder device) from TEXT_ENCODE and TEXT_DEVICE
    (TEXT_DEVICE='cpu' is memory 'cpu'; an unset memory follows `encoder_memory` over
    the memory plan, else 'auto')."""
    spec = {
        **dict(k.get("TEXT_ENCODE") or {}),
        "memory": encoder_memory(k, "text", "auto", plan),
    }
    te_device = torch.device(k["TEXT_DEVICE"] or device)
    if te_device.type == "cpu":
        spec["memory"] = "cpu"
    return text_plan(spec), te_device


def data_knobs(k: dict, device: torch.device) -> dict:
    """The knobs the run's DataConfig reads: `k` plus the encoder compute specs its
    cache keys record (VAE_DTYPE resolved on `device`; the text encoder's resolved
    dtype, precision and quantization layout)."""
    plan, te_device = training_text_plan(k, device)
    on = torch.device("cpu") if plan.memory == "cpu" else te_device
    plan, _, layout = resolve_text_plan(_component_path(k, TEXT), plan, on)
    vae_dtype = resolve_vae_dtype(k.get("VAE_DTYPE", "auto"), device)
    return {**k, **cache_specs(vae_dtype, plan, TEXT_DTYPES[plan.dtype], layout)}


def sample_texts(k: dict) -> list[str]:
    """The fixed prompts sampling encodes: SAMPLE_PROMPTS, then SAMPLE_NEGATIVE."""
    return list(k["SAMPLE_PROMPTS"]) + [k.get("SAMPLE_NEGATIVE", "")]


def build_encoders(
    k: dict, device: torch.device, prec: PrecisionConfig, reserve=0, mem_plan=None
):
    """(VAE, text encoder, sample-prompt embeddings): the VAE when latents are encoded
    online or samples are drawn; the text encoder when text is encoded online. Sample
    prompts are encoded once here; with an offline text cache the encoder is freed
    right after that pass. The VAE is placed by VAE_ENCODE; an unset memory of either
    follows the DiT memory plan `mem_plan` (`encoder_memory`). `reserve` is the DiT
    step's peak the `auto` text-encoder and VAE placements keep free."""
    online_text = k["TEXT_CACHE"] != "offline"
    sampling = k["SAMPLE_EVERY"] > 0
    need_vae = k["LATENT_CACHE"] != "offline" or sampling
    vae = text = embeds = None
    vae_path = _component_path(k, VAE)
    text_path = _component_path(k, TEXT)
    if need_vae and vae_path:
        vae_dtype = resolve_vae_dtype(k.get("VAE_DTYPE", "auto"), device)
        vae = place_vae(
            FluxVae.from_pretrained(vae_path, "cpu", vae_dtype, k["LATENT_DRAW"]),
            {
                **dict(k.get("VAE_ENCODE") or {}),
                "memory": encoder_memory(k, "vae", "resident", mem_plan),
            },
            device,
            reserve,
            int(k["TARGET_AREA"]),
        )
        rule = encoder_rule(k, "vae", mem_plan)
        log.info(f"VAE placement: {hl(vae.plan.memory)} ({rule})")
    if (online_text or sampling) and text_path:
        plan, te_device = training_text_plan(k, device, mem_plan)
        if not online_text:
            plan = replace(plan, free_after_use=True)
        hold = device if plan.ahead_device == "device" else torch.device("cpu")
        choose = partial(default_auto_memory, plan=plan, reserve=reserve)
        w4a8 = partial(default_w4a8, plan=plan, reserve=reserve)
        text = QwenTextEncoder.from_pretrained(
            text_path,
            te_device,
            plan,
            output_device=hold,
            auto_memory=choose,
            auto_w4a8=w4a8,
        )
        rule = encoder_rule(k, "text", mem_plan)
        log.info(f"text encoder placement: {hl(text.plan.memory)} ({rule})")
        if sampling:
            embeds = [t.to(device) for t in text.encode(sample_texts(k))]
        if not online_text:
            text = None
    return vae, text, embeds


def export(k: dict, built: Built, root: Path, metadata: dict) -> list[Path]:
    """Every export file of the knobs (`training.exports.export_files`) under `root`."""
    files = export_files(k, fused_qkv(k))
    return write_exports(
        files, built.trainable, built.backbone, root, run_name(k), metadata
    )
