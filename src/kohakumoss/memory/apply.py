"""Apply a MemoryPlan to model components, once, at build time (training or inference):
quantize the frozen base -> place / stream block weights -> activation offload -> the
BlockRunner. The caller re-binds ops afterwards (build order in types/context.py)."""

from dataclasses import dataclass, field
from functools import partial

import torch
from torch import nn

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.memory.runners import build_runner
from kohakumoss.model.io.quant_source import backbone_source
from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.offload.optim.fused_step import FusedStep
from kohakumoss.offload.optim.grad import GradOffload
from kohakumoss.offload.weights.layers import move_except
from kohakumoss.offload.weights.shared import SharedHost, block_owners
from kohakumoss.offload.weights.stream import WeightStream
from kohakumoss.quant.bind import rebind_quant, work_device
from kohakumoss.quant.compute import COMPUTE as FP8_COMPUTE
from kohakumoss.quant.compute import ComputeWeight
from kohakumoss.quant.convert import (
    attach_compute,
    default_filter,
    quantizable,
    quantize_model,
    unread_problems,
)
from kohakumoss.quant.densify import densify_bases, stored_formats
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_kernels.dist import dist_rank, grouped

QUANT = ("fp8", "int8", "fp8_compute", "mxfp8")
log = get_logger(__name__)


@dataclass
class MemoryEngines:
    """Everything a plan installed; the driver calls the step hooks below."""

    plan: MemoryPlan
    runner: object = None
    offload: ActivationOffload | None = None
    weight_stream: WeightStream | None = None
    fused: FusedStep | None = None
    grads: GradOffload | None = None
    fp8: list = field(default_factory=list)
    reports: dict = field(default_factory=dict)
    backbone: nn.Module | None = None

    @property
    def host_stepped(self) -> bool:
        """Streamed trainable blocks are stepped by a host optimiser after backward."""
        return (
            self.weight_stream is not None
            and self.weight_stream.trainable
            and self.weight_stream.optimizer is None
        )

    def before_step(self) -> None:
        """Land offloaded / streamed-block gradients on the host before optimiser.step."""
        if self.host_stepped:
            self.weight_stream.expose_grads()
        if self.grads is not None:
            self.grads.drain()

    def after_step(self) -> None:
        """After a host-stepped optimizer step: push the updated host masters into the
        streaming copies and end the step's host gradient sums (an in-backward block
        optimiser finishes itself); reset the activation offload and the gradient
        offload; requantize trainable fp8-compute weights."""
        for lin in self.fp8:
            lin.compute.refresh(lin.weight)
        if self.host_stepped:
            self.weight_stream.refresh()
            self.weight_stream.zero_grads()
        if self.offload is not None:
            self.offload.reset()
        if self.grads is not None:
            self.grads.zero()

    def after_restore(self) -> None:
        """Push restored weights into the copies the forward reads: the streaming copies
        of trainable streamed masters, the trainable fp8-compute caches."""
        for lin in self.fp8:
            lin.compute.refresh(lin.weight)
        if self.weight_stream is not None and self.weight_stream.trainable:
            self.weight_stream.refresh()

    def exclude_params(self, params) -> None:
        """Never offload these parameters' saved tensors (adapters bound after the plan);
        the kept set is rebuilt from the live backbone tensors and ring slots, so the
        storages replaced since `apply_to_backbone` drop out of it."""
        if self.offload is None:
            return
        live = []
        if self.backbone is not None:
            live += [*self.backbone.parameters(), *self.backbone.buffers()]
        if self.weight_stream is not None:
            live += self.weight_stream.storages()
        self.offload.rebuild([*live, *params])

    def host_params(self) -> list[nn.Parameter]:
        """The trainable streamed block weights, whose fp32 masters live on the host
        (stepped in backward by StreamedAdamW with a GPU optim_state, by the host AdamW
        with 'cpu'); empty without a trainable stream."""
        if self.weight_stream is None or not self.weight_stream.trainable:
            return []
        return list(self.weight_stream.params.values())

    def close(self) -> None:
        """Stop every host worker thread the engines started."""
        for engine in (self.weight_stream, self.grads):
            if engine is not None:
                engine.close()


def _node_share(trainable: bool, n_blocks: int, device, plan: MemoryPlan) -> dict:
    """WeightStream kwargs when several ranks of one node train the same streamed
    blocks (DDP): one shared host region for the masters and streaming copies, and
    the block owners of the host optimiser shards; none otherwise."""
    if not (trainable and grouped()):
        return {}
    pin = plan.pinned and torch.device(device).type == "cuda"
    rank, world = dist_rank()
    return {
        "share": partial(SharedHost, pin=pin),
        "owners": block_owners(n_blocks, world),
        "rank": rank,
    }


def stored_base_problems(
    stored: str, precision: str | None, trainable_base: bool
) -> list[str]:
    """Why a checkpoint storing `stored` bases cannot run base precision `precision`
    (None / `auto`: the stored format): a full finetune trains dense weights, so `auto`
    on a pre-quantized file is refused naming the `compute` escape."""
    if stored == "compute" or not trainable_base or precision not in (None, "auto"):
        return []
    return [
        f"the DiT checkpoint stores {stored} bases and a full finetune trains dense "
        "weights: set BASE_PRECISION='compute' to train its dequantized weights, or "
        "use the 16-bit file"
    ]


def settle_stored(backbone: nn.Module, precision: str, trainable_base: bool) -> None:
    """Dequantize the attached stored bases unless `precision` keeps them (`auto` or
    their own format, frozen); ValueError for `auto` with a trainable base."""
    formats = stored_formats(backbone)
    if not formats:
        return
    stored = formats.pop() if len(formats) == 1 else "mixed"
    problems = stored_base_problems(stored, precision, trainable_base)
    if problems:
        raise ValueError(problems[0])
    if not trainable_base and precision in ("auto", stored):
        return
    count = densify_bases(backbone)
    log.info(
        f"base_precision {hl(precision)}: {hl(count)} stored {hl(stored)} bases "
        "dequantized to dense weights (W = dequant(q) with any rotation undone)"
    )


def quantized_weights(
    backbone: nn.Module,
    plan: MemoryPlan,
    trainable_base: bool,
    keep_fp32: tuple[str, ...] = (),
    quant_min_features: int = 256,
) -> set[str]:
    """The weight names `quantize_base` replaces under `plan` (none for a trainable base
    or a precision it does not quantize): what a load may leave unread
    (`load_backbone(defer=...)`); works on the meta model."""
    if plan.base_precision not in QUANT or trainable_base:
        return set()
    accept = default_filter(keep_fp32, quant_min_features)
    linears, _ = quantizable(backbone, plan.base_precision, accept)
    return {f"{name}.weight" for name, _ in linears}


@torch.no_grad()
def quantize_base(
    backbone: nn.Module,
    plan: MemoryPlan,
    compute_dtype: torch.dtype,
    training: bool,
    trainable_base: bool,
    keep_fp32: tuple[str, ...] = (),
    quant_min_features: int = 256,
    work: torch.device | str | None = None,
) -> dict | None:
    """The frozen base's quantized storage under `plan.base_precision` (quantized on
    `work` from the checkpoint's stored values where they apply, `backbone_source`;
    the meta device builds the layout only), with the fp8 input-gradient copy when
    training takes it; the quantization report, None when nothing is quantized. Bases
    a pre-quantized checkpoint attached stay under `auto` and their own format, and are
    dequantized first for any other precision (`compute` included) and for a
    trainable base, which `auto` refuses (`stored_base_problems`)."""
    report = None
    settle_stored(backbone, plan.base_precision, trainable_base)
    if plan.base_precision in QUANT and not trainable_base:
        source, scales = backbone_source(backbone)
        report = quantize_model(
            backbone,
            plan.base_precision,
            default_filter(keep_fp32, quant_min_features),
            compute_dtype,
            recipe=plan.fp8_recipe,
            dgrad=training and plan.fp8_dgrad == "fp8",
            work_device=work,
            source=source,
            scales=scales,
        )
    problems = unread_problems(backbone)
    if problems:
        raise ValueError(problems[0])
    if training and plan.fp8_dgrad == "fp8":
        for m in backbone.modules():
            if isinstance(m, ComputeWeight):
                m.add_dgrad()
    return report


@torch.no_grad()
def trainable_compute(
    backbone: nn.Module,
    plan: MemoryPlan,
    compute_dtype: torch.dtype,
    training: bool,
    trainable_base: bool,
    keep_fp32: tuple[str, ...] = (),
    quant_min_features: int = 256,
) -> list | None:
    """fp8 compute caches on the trainable dense linears (fp8_compute / mxfp8 with a
    trainable base, in training); the linears, None when the plan takes none."""
    if not (plan.base_precision in FP8_COMPUTE and trainable_base and training):
        return None
    return attach_compute(
        backbone,
        plan.base_precision,
        default_filter(keep_fp32, quant_min_features),
        compute_dtype,
        plan.fp8_recipe,
        plan.fp8_dgrad == "fp8",
    )


@torch.no_grad()
def apply_to_backbone(
    backbone: nn.Module,
    plan: MemoryPlan,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    training: bool = True,
    trainable_base: bool = False,
    keep_fp32: tuple[str, ...] = (),
    quant_min_features: int = 256,
) -> MemoryEngines:
    """Install quantization, streaming, activation offload and the runner on `backbone`
    (anything exposing `layers` and `set_runner`), on any device: weights end on
    `device` except the streamed block weights, which stay in host memory."""
    engines = MemoryEngines(plan, backbone=backbone)
    report = quantize_base(
        backbone,
        plan,
        compute_dtype,
        training,
        trainable_base,
        keep_fp32,
        quant_min_features,
        work_device(device),
    )
    if report is not None:
        engines.reports["quant"] = report
    if plan.stream == "off":
        backbone.to(device)
    rebind_quant(backbone, device, compute_dtype, autograd=training)
    fp8 = trainable_compute(
        backbone,
        plan,
        compute_dtype,
        training,
        trainable_base,
        keep_fp32,
        quant_min_features,
    )
    if fp8 is not None:
        engines.fp8 = fp8
        engines.reports["quant"] = {
            "layers": len(engines.fp8),
            "cache_bytes": sum(lin.compute.nbytes() for lin in engines.fp8),
        }
    if plan.stream != "off":
        move_except(backbone, backbone.layers, device)
        trainable = plan.stream == "swap" and trainable_base
        engines.weight_stream = WeightStream(
            backbone.layers,
            device,
            slots=max(plan.stream_resident, 2),
            pinned=plan.pinned,
            trainable=trainable,
            dtype=compute_dtype,
            **_node_share(trainable and training, len(backbone.layers), device, plan),
        )
        masters = {id(p) for p in engines.weight_stream.params.values()}
        for block in backbone.layers:
            for t in list(block.parameters()) + list(block.buffers()):
                if id(t) not in masters:
                    t.data = t.data.to(device)
    if training and plan.act_offload:
        engines.offload = ActivationOffload(
            device, pinned=plan.pinned, inflight=plan.act_offload_inflight
        )
        engines.offload.exclude(p for p in backbone.parameters())
        engines.offload.exclude(p for p in backbone.buffers())
        if engines.weight_stream is not None:
            engines.offload.exclude(engines.weight_stream.storages())
    engines.runner = build_runner(
        plan.checkpoint if training else "none",
        backbone.layers,
        offload=engines.offload,
        weight_stream=engines.weight_stream,
    )
    backbone.set_runner(engines.runner)
    return engines


@torch.no_grad()
def apply_inference_plan(
    plan: MemoryPlan,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    backbone: nn.Module | None = None,
    vae: nn.Module | None = None,
    keep_fp32: tuple[str, ...] = (),
) -> MemoryEngines:
    """The pipeline's DiT / VAE placement under `plan` (the text encoder is placed by
    its own TextEncoderPlan through `apply_to_text_encoder`)."""
    engines = MemoryEngines(plan)
    if backbone is not None:
        engines = apply_to_backbone(
            backbone, plan, device, compute_dtype, training=False, keep_fp32=keep_fp32
        )
    if vae is not None:
        vae.to(device)
    return engines


def attach_grad_offload(engines: MemoryEngines, params, pinned: bool = True):
    """Install gradient offload (host fp32 accumulation across micro-batches) for the
    device-resident trainable `params` (plan.grad_offload). Streamed `swap` blocks
    already accumulate on the host and must not be passed here."""
    engines.grads = GradOffload(params, pinned=pinned)
    return engines.grads


def attach_fused_step(
    engines: MemoryEngines, params, step_param, clip: float = 0.0, scaled: bool = False
) -> FusedStep:
    """Install the in-backward optimiser step for `params` (plan.fused_step);
    `scaled` when a loss scaler is active (then pass `inv_scale` to `start`)."""
    engines.fused = FusedStep(params, step_param, clip, scaled)
    return engines.fused
