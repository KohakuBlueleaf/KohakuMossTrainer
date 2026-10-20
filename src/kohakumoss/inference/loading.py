"""MossPipeline component loaders: the DiT (+ adapters, fp16 fix, memory plan), the text
encoder and the VAE, each placed for the device; `component_root` finds their folders.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path

import torch

from kohakumoss import adapters as adapter_api
from kohakumoss.encoders import FluxVae, QwenTextEncoder
from kohakumoss.encoders.text import text_plan as make_text_plan
from kohakumoss.encoders.text.encoder import default_auto_memory, default_w4a8
from kohakumoss.encoders.vae import resolve_vae_dtype
from kohakumoss.inference.placement_report import REFERENCE_TOKENS
from kohakumoss.logging.log import get_logger, hl, log_object, timed
from kohakumoss.memory.apply import apply_inference_plan, settle_stored
from kohakumoss.memory.estimate import Workload, estimate_inference
from kohakumoss.memory.profiles import ORDER
from kohakumoss.memory.profiles import resolve as resolve_memory
from kohakumoss.model.build import DEFAULT_KEEP_FP32
from kohakumoss.model.io import load_backbone
from kohakumoss.model.numerics import apply_fp16_fix, output_row_scales
from kohakumoss.model.opset import OpSet
from kohakumoss.offload.accel import device_budget
from kohakumoss.offload.encoders.vae import place_vae
from kohakumoss.quant.densify import stored_precision_of
from kohakumoss.types.arch import MossArchConfig
from kohakumoss.types.plans import MemoryPlan, TextEncoderPlan
from kohakumoss.weights import (
    DIT,
    TEXT,
    VAE,
    component_path,
    component_root,
    is_comfy_folder,
)
from kohakumoss_common.units import GIB
from kohakumoss_kernels.dtypes import short

log = get_logger(__name__)
VAE_PIXELS = 1024 * 1024
AUTO = "auto"


def auto_profile(
    arch: MossArchConfig,
    compute_bytes: int,
    device,
    overrides: dict | None,
    budget: int,
    tokens: int = REFERENCE_TOKENS,
    stored: str = "compute",
) -> str:
    """MEMORY_PROFILE 'auto' for sampling: the fastest profile (ORDER) whose inference
    estimate of one `tokens`-token forward of `arch` at `compute_bytes` per element
    (the DiT file's `stored` precision under base precision auto) fits `budget` bytes;
    ValueError listing each estimate when none fits."""
    work = Workload(
        arch=arch,
        img_tokens=tokens,
        txt_tokens=0,
        compute_bytes=compute_bytes,
        stored_precision=stored,
    )
    rows = []
    for name in ORDER:
        plan = resolve_memory(name, overrides, device, inference=True)
        need = estimate_inference(work, plan).total
        if need <= budget:
            return name
        rows.append(f"{name} {need / GIB:.2f} GiB")
    raise ValueError(
        f"MEMORY_PROFILE='auto': no profile fits {budget / GIB:.2f} GiB "
        f"({'; '.join(rows)}); pick a MEMORY_PROFILE or a quantized BASE_PRECISION"
    )


@dataclass
class LoadedDit:
    """The DiT as `load_dit` placed it: backbone, adapter stacks, the inference memory
    plan (empty = resident) and its weight stream (None = resident)."""

    backbone: torch.nn.Module
    stacks: list = field(default_factory=list)
    plan: MemoryPlan = field(default_factory=MemoryPlan)
    weight_stream: object = None


def _plan_dict(plan) -> dict:
    if plan is None:
        return {}
    return asdict(plan) if isinstance(plan, TextEncoderPlan) else dict(plan)


def component_paths(
    path: Path,
    variant: str | None,
    transformer: str = "",
    text_encoder: str = "",
    vae: str = "",
) -> tuple[Path, Callable[[], Path], Callable[[], Path]]:
    """(DiT path, text-encoder path, VAE path) of a weights `path`: a ComfyUI models
    folder (`weights.component_path`: each knob a file name there, a path, or "" for
    the default), else the HF layout (`component_root`; a set knob is a path). The
    encoder paths are resolved when called (a run may load neither)."""
    if is_comfy_folder(path):
        dit = component_path(path, DIT, transformer, variant or "base")
        return (
            dit,
            partial(component_path, path, TEXT, text_encoder, variant or "base"),
            partial(component_path, path, VAE, vae, variant or "base"),
        )

    def hf(name: str, choice: str) -> Path:
        return Path(choice) if choice else component_root(path, variant, name)

    dit = Path(transformer) if transformer else component_root(path, variant, DIT)
    return dit, partial(hf, TEXT, text_encoder), partial(hf, VAE, vae)


def load_transformer(
    dit: str | Path,
    dtype: torch.dtype,
    device: torch.device | str,
    adapters: Sequence[tuple[str, float]] = (),
    adapter_mode: str = "merge",
    fp16_fix: str = "none",
    fp16_fix_kwargs: dict | None = None,
    kernels: dict[str, str] | None = None,
    base_precision: str = "auto",
):
    """Backbone + adapters (merged before a weight fix, bypassed after it) + fp16 fix,
    ops bound last; adapter deltas always carry the row scales of a fold already in
    the weights (an fp16-fix file). A pre-quantized DiT file keeps its stored bases
    under `base_precision` auto / their own format and is dequantized for any other
    (`settle_stored`) before the adapters attach. Returns (backbone, adapter
    stacks)."""
    fix_kw = dict(fp16_fix_kwargs or {})
    backbone, _ = load_backbone(
        dit,
        device=device,
        dtype=dtype,
        kernels=kernels,
        fp16_fix_eps=fix_kw.get("eps", "stock"),
    )
    backbone.eval().requires_grad_(False)
    settle_stored(backbone, base_precision, trainable_base=False)
    stacks = []
    if adapter_mode == "merge":
        loaded = output_row_scales(backbone)
        for file, mult in adapters:
            stacks.append(
                adapter_api.load(backbone, file, mult, mode="merge", row_scales=loaded)
            )
        apply_fp16_fix(backbone, fp16_fix, **fix_kw)
    else:
        apply_fp16_fix(backbone, fp16_fix, **fix_kw)
        scales = output_row_scales(backbone)
        for file, mult in adapters:
            stacks.append(
                adapter_api.load(backbone, file, mult, mode="bypass", row_scales=scales)
            )
    backbone.requires_grad_(False)
    backbone.bind_ops()
    return backbone, stacks


def attach_after_plan(backbone, adapters, adapter_mode: str, device) -> list:
    """Adapters loaded after a memory plan placed the DiT, so each layer sees its final
    base: a quantized / streamed one attaches as bypass whatever `adapter_mode`
    (logged once by the stack); deltas carry the row scales of a fold in the weights."""
    scales = output_row_scales(backbone)
    stacks = [
        adapter_api.load(
            backbone, file, mult, mode=adapter_mode, row_scales=scales, device=device
        )
        for file, mult in adapters
    ]
    backbone.requires_grad_(False)
    return stacks


def load_dit(
    dit: Path,
    dtype: torch.dtype,
    device: torch.device | str,
    adapters: Sequence[tuple[str, float]],
    adapter_mode: str,
    fp16_fix: str,
    fp16_fix_kwargs: dict | None,
    kernels: dict[str, str] | None,
    memory: str | None,
    memory_overrides: dict | None,
    tokens: int = REFERENCE_TOKENS,
) -> LoadedDit:
    """The DiT on `device`: resident with adapters attached by `load_transformer`, or
    (a `memory` profile) loaded on the host, placed by the inference MemoryPlan, then
    the adapters attached to the placed layers; 'auto' sizes for a `tokens`-token
    forward."""
    precision = (memory_overrides or {}).get("base_precision", "auto")
    if not memory:
        backbone, stacks = load_transformer(
            dit,
            dtype,
            device,
            adapters,
            adapter_mode,
            fp16_fix,
            fp16_fix_kwargs,
            kernels,
            precision,
        )
        loaded = LoadedDit(backbone, stacks)
    else:
        backbone, _ = load_transformer(
            dit, dtype, "cpu", (), adapter_mode, fp16_fix, fp16_fix_kwargs, kernels
        )
        if memory == AUTO:
            budget = device_budget(device)
            cb = backbone.final_norm_video.weight.element_size()
            memory = auto_profile(
                backbone.cfg,
                cb,
                device,
                memory_overrides,
                budget,
                tokens,
                stored_precision_of(backbone),
            )
            log.info(f"MEMORY_PROFILE 'auto' -> {hl(memory)}")
        plan = resolve_memory(memory, memory_overrides, device, inference=True)
        began = time.perf_counter()
        weight_stream = apply_inference_plan(
            plan, device, dtype, backbone=backbone, keep_fp32=DEFAULT_KEEP_FP32
        ).weight_stream
        backbone.bind_ops(OpSet.resolve(device, dtype, kernels))
        took = f"{time.perf_counter() - began:.1f}s"
        suffix = f" on {device} (applied in {hl(took)})"
        log_object(log, "DiT memory plan", plan, suffix=suffix)
        stacks = attach_after_plan(backbone, adapters, adapter_mode, device)
        loaded = LoadedDit(backbone, stacks, plan, weight_stream)
    if adapters:
        modes = sorted({s.effective_mode for s in loaded.stacks}) or [adapter_mode]
        names = [f"{a}:{m}" for a, m in adapters]
        log.info(f"adapters ({', '.join(modes)}): {names}")
    return loaded


def load_text_encoder(
    root: Path,
    device: torch.device | str,
    text_device: torch.device | str | None,
    text_plan: TextEncoderPlan | dict | None,
    reserve: int,
) -> QwenTextEncoder:
    """The text encoder computing on `text_device` (default `device`; "cpu" = memory
    'cpu') in the dtype `device` resolves 'auto' to, its outputs on `device`; an unset
    plan memory is 'auto' keeping `reserve` bytes free."""
    te_device = torch.device(text_device or device)
    spec = make_text_plan({"memory": "auto", **_plan_dict(text_plan)})
    if te_device.type == "cpu":
        spec = replace(spec, memory="cpu")
    choose = partial(default_auto_memory, plan=spec, reserve=reserve)
    w4a8 = partial(default_w4a8, plan=spec, reserve=reserve)
    what = (
        f"text encoder streamed to its placement (memory {spec.memory}, compute "
        f"on {te_device})"
    )
    with timed(log, what) as done:
        te = QwenTextEncoder.from_pretrained(
            root,
            te_device,
            spec,
            output_device=device,
            auto_memory=choose,
            auto_w4a8=w4a8,
            run_device=device,
        )
        done["memory"] = te.plan.memory
    return te


def load_vae(
    root: Path,
    vae_dtype: torch.dtype | str,
    device: torch.device | str,
    vae_plan: dict | None,
    reserve: int,
):
    """The VAE read to the host in `vae_dtype` ("auto" resolved for `device`), placed
    by `vae_plan` (VaePlan fields; an unset memory is 'auto' keeping `reserve` bytes
    free at a 1024x1024 decode)."""
    vae_dtype = resolve_vae_dtype(vae_dtype, device)
    with timed(log, f"VAE weights -> host RAM ({short(vae_dtype)})"):
        decoder = FluxVae.from_pretrained(root, "cpu", vae_dtype)
    spec = {"memory": "auto", **(vae_plan or {})}
    return place_vae(decoder, spec, device, reserve, VAE_PIXELS)
