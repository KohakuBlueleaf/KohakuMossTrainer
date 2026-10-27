"""Resolved run settings: precision config, memory plan, parallel plan (built once at
startup)."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PrecisionConfig:
    """Compute dtype, modules kept in fp32, loss scaling and the residual fold."""

    compute: torch.dtype = torch.bfloat16
    master: torch.dtype = torch.float32
    keep_fp32: tuple[str, ...] = (
        "time_embedding",
        "shared_block_modulation",
        "video_head_modulation",
    )
    scaler: bool = False
    scaler_init: float = 2.0**16
    scaler_growth_interval: int = 2000
    fp16_fix: str = "none"
    fp16_fix_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPlan:
    """Concrete memory/speed knobs a profile resolves to; every field is overridable.
    `base_precision` auto keeps the checkpoint's stored format (16-bit: compute; a
    pre-quantized file: its bases); compute is 16-bit (a pre-quantized file
    dequantized); int8 / fp8 weight-only; fp8_compute / mxfp8 run the frozen base's
    GEMMs in fp8 (`fp8_recipe` tensor | row | block for fp8_compute); `fp8_dgrad` fp8
    stores a W^T copy for an fp8 input-gradient GEMM, 16bit dequantizes the forward
    copy. `int8_compute` runs every int8 base's GEMMs in int8 (activations quantized
    per row, the `int8_linear` op) instead of weight-only; `int8_dgrad` int8 | 16bit
    is its input gradient.
    `act_offload_inflight`: blocks whose activation-offload copies may be queued before
    the host waits for the oldest block's."""

    profile: str = "balanced"
    kernels: dict = field(default_factory=dict)
    checkpoint: str = "block"
    act_offload: bool = False
    act_offload_inflight: int = 2
    base_precision: str = "auto"
    fp8_recipe: str = "row"
    fp8_dgrad: str = "fp8"
    int8_compute: bool = False
    int8_dgrad: str = "int8"
    stream: str = "off"
    stream_resident: int = 0
    pinned: bool = True
    optim_state: str = "gpu"
    grad_offload: bool = False
    fused_step: bool = False
    compile: str = "off"


@dataclass(frozen=True)
class TextEncoderPlan:
    """Text-encoder placement, precision and batching (training and inference).

    memory: resident | offload (whole model to the device per pass) | ring (a layer
    ring of `stream_resident` slots, prefetching; "stream" is its alias) | cpu (runs on
    the host) | auto (resolved by the memory planner); `free_after_use` drops the model
    after its first pass. precision: auto (the format the weights store: compute for a
    16-bit checkpoint, a pre-quantized file's own; compute on the host) | compute |
    int8 | fp8 (weight-only, per output channel or per `quant_group` input columns) |
    fp8_compute (fp8 GEMMs, `fp8_recipe` tensor | row | block scaling, sm89+) | mxfp8
    (MX block-scaled GEMMs, sm89+) | w4a8 (stored only: a ComfyUI W4A8 file); fp8
    compute weights are quantized once, activations per call. `w4a8` decodes a W4A8
    file's codes: int8 (once, 1 byte per weight) | packed (per call) | auto (int8 when
    the decoded encoder fits resident next to the DiT step). `int8_compute` runs the
    int8 linears' GEMMs in int8 (activations quantized per row; CUDA) instead of
    weight-only. `dtype` bf16 | fp16 |
    auto (bf16, or fp16 where the device has no bf16). `batch` prompts / `max_tokens`
    tokens per kernel call, `layout`
    single | padded | packed, `forward` kernels (kohakumoss_kernels ops) | hf (the
    transformers modules' own forward; single / padded only). `ahead` host batches
    encoded per pass by the online text cache, held on `ahead_device` host | device.
    `kernels` maps a text op to a backend. `stored` is a pre-quantized text-encoder
    file (scripts/export/text_quant.py) of the same precision: its quantized linears
    are attached instead of quantizing at load. `host`: the host form of the weights
    offload / ring keep on the host: copy (copies in the compute dtype, pinned when
    `pinned`) | file (the checkpoint's own file pages, cast by the copy to the device)
    | auto (copy when it fits the host's free memory, else file). See
    docs/internals/text-encoder.md."""

    memory: str = "resident"
    free_after_use: bool = False
    precision: str = "auto"
    w4a8: str = "auto"
    int8_compute: bool = False
    quant_group: int = 0
    fp8_recipe: str = "row"
    stored: str = ""
    dtype: str = "auto"
    stream_resident: int = 2
    pinned: bool = True
    host: str = "auto"
    batch: int = 32
    max_tokens: int = 16384
    layout: str = "packed"
    forward: str = "kernels"
    ahead: int = 1
    ahead_device: str = "host"
    kernels: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelPlan:
    """Multi-device layout and process launch: `timeout_s` is every collective's and
    the store's wait (DIST_TIMEOUT), `bind` the process group's device binding
    (DIST_BIND: "lazy" | "device")."""

    kind: str = "none"
    backend: str = "auto"
    init: str = "tcp"
    devices: tuple[int, ...] = (0,)
    stages: tuple[int, ...] = ()
    schedule: str = "1f1b"
    microbatches: int = 1
    timeout_s: float = 600.0
    bind: str = "lazy"
