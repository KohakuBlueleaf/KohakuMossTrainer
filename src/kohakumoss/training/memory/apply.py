"""Apply a MemoryPlan to model components, once, at build time (training or inference).

Order on the backbone: quantize the frozen base -> stream block weights -> activation
offload -> compose the BlockRunner. The caller re-binds ops afterwards
(`backbone.bind_ops()`, build order in types/api.py).
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.offload.fused_step import FusedStep
from kohakumoss.offload.grad import GradOffload
from kohakumoss.offload.layers import LayerStreamer, find_layer_list, move_except
from kohakumoss.offload.stream import WeightStream
from kohakumoss.quant.convert import (
    default_filter,
    quantize_model,
    quantize_torch_linears,
)
from kohakumoss.quant.weights import QuantWeight
from kohakumoss.training.memory.runners import build_runner
from kohakumoss.types.plans import MemoryPlan

QUANT = ("fp8", "int8")


@dataclass
class MemoryEngines:
    """Everything a plan installed; the driver calls the step hooks below."""

    plan: MemoryPlan
    runner: object = None
    offload: ActivationOffload | None = None
    stream: WeightStream | None = None
    text_stream: LayerStreamer | None = None
    fused: FusedStep | None = None
    grads: GradOffload | None = None
    reports: dict = field(default_factory=dict)

    def before_step(self) -> None:
        """Land offloaded / streamed-block gradients on the host before optimiser.step."""
        if self.stream is not None and self.stream.trainable:
            self.stream.flush_grads()
        if self.grads is not None:
            self.grads.flush()

    def after_step(self) -> None:
        """Push updated host masters into the streaming copies, clear host grads."""
        if self.stream is not None and self.stream.trainable:
            if self.stream.optimizer is None:
                self.stream.refresh()
                self.stream.zero_grads()
            else:
                self.stream.optimizer.finish()
        if self.offload is not None:
            self.offload.reset()
        if self.grads is not None:
            self.grads.zero()

    def host_params(self) -> list[nn.Parameter]:
        """Parameters whose masters live on the host (optimise with a host optimiser)."""
        if self.stream is None or not self.stream.trainable:
            return []
        return list(self.stream.params.values())


def _rebind_quant(module: nn.Module, device, dtype) -> None:
    for m in module.modules():
        if isinstance(m, QuantWeight):
            m.bind(dtype, device)


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
    (anything exposing `layers` and `set_runner`). The backbone is already on `device`
    except for blocks the stream will own."""
    engines = MemoryEngines(plan)
    if plan.base_precision in QUANT and not trainable_base:
        accept = default_filter(keep_fp32, quant_min_features)
        engines.reports["quant"] = quantize_model(
            backbone, plan.base_precision, accept, compute_dtype
        )
        _rebind_quant(backbone, device, compute_dtype)
    if plan.stream != "off":
        move_except(backbone, backbone.layers, device)
        engines.stream = WeightStream(
            backbone.layers,
            device,
            resident=max(plan.stream_resident, 2),
            pinned=plan.pinned,
            trainable=plan.stream == "swap" and trainable_base,
            dtype=compute_dtype,
        )
        masters = {id(p) for p in engines.stream.params.values()}
        for block in backbone.layers:
            for t in list(block.parameters()) + list(block.buffers()):
                if id(t) not in masters:
                    t.data = t.data.to(device)
    if training and plan.act_offload:
        engines.offload = ActivationOffload(device, pinned=plan.pinned)
        engines.offload.exclude(p for p in backbone.parameters())
        engines.offload.exclude(p for p in backbone.buffers())
        if engines.stream is not None:
            engines.offload.exclude(engines.stream.storages())
    engines.runner = build_runner(
        plan.checkpoint if training else "none",
        backbone.layers,
        offload=engines.offload,
        engine=engines.stream,
    )
    backbone.set_runner(engines.runner)
    return engines


@torch.no_grad()
def apply_to_text_encoder(
    encoder: nn.Module,
    plan: MemoryPlan,
    device: torch.device | str,
    compute_dtype: torch.dtype,
) -> LayerStreamer | None:
    """Quantize (int8/fp8) and/or layer-stream a frozen text encoder for encoding."""
    if plan.base_precision in QUANT:
        quantize_torch_linears(encoder, plan.base_precision, compute_dtype)
    if plan.stream == "off":
        encoder.to(device)
        _rebind_quant(encoder, device, compute_dtype)
        return None
    layers = find_layer_list(encoder)
    move_except(encoder, layers, device)
    streamer = LayerStreamer(
        layers, device, resident=max(plan.stream_resident, 2), pinned=plan.pinned
    )
    _rebind_quant(encoder, device, compute_dtype)
    return streamer


@torch.no_grad()
def apply_inference_plan(
    plan: MemoryPlan,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    backbone: nn.Module | None = None,
    text_encoder: nn.Module | None = None,
    vae: nn.Module | None = None,
    keep_fp32: tuple[str, ...] = (),
) -> MemoryEngines:
    """The pipeline's single entry: place DiT, text encoder and VAE under `plan`."""
    engines = MemoryEngines(plan)
    if backbone is not None:
        engines = apply_to_backbone(
            backbone, plan, device, compute_dtype, training=False, keep_fp32=keep_fp32
        )
        if plan.stream == "off":
            backbone.to(device)
            _rebind_quant(backbone, device, compute_dtype)
    if text_encoder is not None:
        engines.text_stream = apply_to_text_encoder(
            text_encoder, plan, device, compute_dtype
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
    `scaled` when a loss scaler is active (then pass `inv_scale` to `arm`)."""
    engines.fused = FusedStep(params, step_param, clip, scaled)
    return engines.fused
