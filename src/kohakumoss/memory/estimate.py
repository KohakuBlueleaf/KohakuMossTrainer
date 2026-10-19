"""Analytic GPU-memory model of one training step: weights, trainable state, activations.

The constants are per-token saved-activation counts of the MOSS joint block; see
docs/internals/offload.md for their derivation and the probe that calibrates them.
"""

from dataclasses import dataclass, field

from kohakumoss.memory.profiles import checkpoint_mode
from kohakumoss.offload.optim.gpu_adamw import state_ring_bytes
from kohakumoss.offload.optim.grad import grad_ring_bytes
from kohakumoss.offload.optim.stream_optim import slot_bytes
from kohakumoss.quant.compute import COMPUTE as FP8_COMPUTE
from kohakumoss.types.arch import MossArchConfig
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_common.units import GIB

PRECISION_BYTES = {
    "compute": None,
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp8": 1,
    "int8": 1,
    "fp8_compute": 1,
    "mxfp8": 1 + 1 / 32,
    "w4a8": 0.5 + 1 / 16,
}
QUANTIZED = ("fp8", "int8", "w4a8", *FP8_COMPUTE)
HOST_STATES = ("stream", "cpu")
CONTEXT_BYTES = int(0.88 * GIB)  # device memory outside the torch allocator
WORKSPACE_BYTES = int(0.25 * GIB)
FRAGMENTATION = 1.08  # reserved over allocated, every buffer on the compute stream
FRAGMENTATION_OFFLOAD = 1.12  # reserved over allocated with side-stream buffers
ACT_FACTOR = {"fused": 1.0, "eager": 3.6}


@dataclass(frozen=True)
class Workload:
    """What one rank runs per micro-step. `sharded`: parameters sharded over `world`
    ranks (FSDP2); `sharded_grads`: their gradients sharded too (False while FSDP2
    accumulates unsharded gradients across micro-batches); `gathered`: every sharded
    block also held unsharded from its forward through backward (FSDP_RESHARD=
    'backward', ZeRO-2); `ema_decays`: fp32 EMA
    averages kept of every trainable parameter (on its device; `swap` block masters
    and their averages live on the host); `master_bytes`: a full finetune's trainable
    device parameter (fp32 master, the compute dtype under adamw16);
    `stored_precision`: the base precision the checkpoint stores (what a plan's
    `auto` keeps; `model.io.load.stored_precision`)."""

    arch: MossArchConfig
    img_tokens: int
    txt_tokens: int
    trainable_params: int = 0
    full_ft: bool = False
    compute_bytes: int = 2
    world: int = 1
    sharded: bool = False
    sharded_grads: bool = True
    gathered: bool = False
    act_factor: float = ACT_FACTOR["fused"]
    ema_decays: int = 0
    optim_state_bytes: int = 8
    master_bytes: int = 4
    stored_precision: str = "compute"

    @property
    def tokens(self) -> int:
        return self.img_tokens + self.txt_tokens


def base_precision(plan: MemoryPlan, stored: str = "compute") -> str:
    """The base precision a plan runs: the checkpoint's `stored` one under `auto`."""
    return stored if plan.base_precision == "auto" else plan.base_precision


def fragmentation(plan: MemoryPlan) -> float:
    """The plan's reserved-over-allocated factor: FRAGMENTATION_OFFLOAD when it keeps
    buffers on side streams (streamed weights, host optimiser states, gradient or
    activation offload), else FRAGMENTATION."""
    offload = plan.stream != "off" or plan.act_offload or plan.grad_offload
    if offload or plan.optim_state != "gpu":
        return FRAGMENTATION_OFFLOAD
    return FRAGMENTATION


def state_bytes(work: Workload, plan: MemoryPlan) -> int:
    """Device optimiser-state bytes per trainable parameter: the OPTIMIZER's
    (`optim_state_bytes`) with GPU states, none with host states."""
    return 0 if plan.optim_state in HOST_STATES else work.optim_state_bytes


@dataclass
class Estimate:
    """Bytes per component and their device total: the allocator's parts times
    `fragmentation` (reserved over allocated), plus the `context` part (outside the
    allocator)."""

    parts: dict[str, int] = field(default_factory=dict)
    fragmentation: float = FRAGMENTATION

    def device(self, names=None) -> int:
        """Device bytes of the parts `names` (default: every part)."""
        names = list(self.parts) if names is None else names
        torch_bytes = sum(self.parts.get(k, 0) for k in names if k != "context")
        context = self.parts.get("context", 0) if "context" in names else 0
        return int(torch_bytes * self.fragmentation) + context

    @property
    def total(self) -> int:
        return self.device()

    def gib(self) -> dict[str, float]:
        out = {k: round(v / GIB, 3) for k, v in self.parts.items()}
        out["total"] = round(self.total / GIB, 3)
        return out


def block_params(arch: MossArchConfig) -> int:
    """Parameters of one joint block."""
    d, f = arch.dim, arch.ffn_dim
    return 4 * d * d + 3 * d * f + 2 * d + 2 * arch.head_dim


def text_block_params(arch: MossArchConfig) -> int:
    """Parameters of one text-refiner block (biased)."""
    d, f = arch.dim, arch.ffn_dim
    return 4 * d * d + 4 * d + 3 * d * f + 2 * f + d + 2 * d + 2 * arch.head_dim


def other_params(arch: MossArchConfig) -> int:
    """Embedders, time MLP, modulations and head."""
    d, c = arch.dim, arch.latent_channels
    return (
        arch.text_dim * d
        + d
        + c * d
        + d
        + arch.freq_dim * d
        + d
        + d * d
        + d
        + 6 * d * d
        + 6 * d
        + 2 * d * d
        + 2 * d
        + d * c
        + d
    )


def largest_device_param(arch: MossArchConfig, swap: bool) -> int:
    """Elements of a full finetune's largest device parameter: the block-sharing
    modulation (6 D x D), the text projection or, without `swap` (blocks on the device),
    a block's fused FFN up / gate (2 F x D) or q/k/v (3 D x D)."""
    d, f = arch.dim, arch.ffn_dim
    outside = max(6 * d * d, arch.text_dim * d, arch.freq_dim * d)
    return outside if swap else max(outside, 2 * f * d, 3 * d * d)


def total_params(arch: MossArchConfig) -> int:
    return (
        arch.depth * block_params(arch)
        + arch.text_blocks * text_block_params(arch)
        + other_params(arch)
    )


def saved_per_token(arch: MossArchConfig, mode: str) -> int:
    """Saved activation elements per token per block for a checkpoint mode."""
    d, f = arch.dim, arch.ffn_dim
    full = 12 * d + 3 * f
    match mode:
        case "none":
            return full
        case "block":
            return d
        case "sub-block":
            return 2 * d
        case "selective":
            return 6 * d + 2 * f
        case _:
            raise ValueError(mode)


def recompute_window(arch: MossArchConfig, mode: str) -> int:
    """Extra elements per token alive while one checkpointed region recomputes."""
    d, f = arch.dim, arch.ffn_dim
    full = 12 * d + 3 * f
    return {"none": 0, "block": full, "sub-block": full // 2, "selective": 6 * d + f}[
        mode
    ]


def weight_bytes(
    plan: MemoryPlan,
    compute_bytes: int,
    training: bool,
    full_ft: bool = False,
    stored: str = "compute",
) -> float:
    """Stored bytes per base parameter: the compute dtype, a quantized format (the
    checkpoint's `stored` one under `auto`), plus the W^T copy of an fp8 input gradient
    in training; a trainable fp8-compute weight keeps its 16-bit copy next to the fp8
    ones."""
    precision = base_precision(plan, stored)
    wb = PRECISION_BYTES.get(precision) or compute_bytes
    fp8 = precision in FP8_COMPUTE
    copies = 2 if training and fp8 and plan.fp8_dgrad == "fp8" else 1
    return wb * copies + (compute_bytes if fp8 and full_ft else 0)


def _transients(est: Estimate, work: Workload, plan: MemoryPlan) -> None:
    """Per-call buffers of a quantized base: a dequantized layer (weight-only, and the
    fp8 path's 16-bit fallback) and the fp8 activation copy."""
    arch, cb = work.arch, work.compute_bytes
    precision = base_precision(plan, work.stored_precision)
    if precision not in QUANTIZED:
        return
    largest = 2 * arch.ffn_dim * arch.dim
    est.parts["dequant"] = (block_params(arch) + 2 * largest) * cb
    if precision in FP8_COMPUTE:
        est.parts["fp8_act"] = work.tokens * 2 * arch.ffn_dim


def estimate_inference(work: Workload, plan: MemoryPlan) -> Estimate:
    """Peak GPU bytes of one no-grad DiT forward under `plan`: the placed weights plus
    one block's live activations (nothing is saved for backward)."""
    arch, cb = work.arch, work.compute_bytes
    est = Estimate(fragmentation=fragmentation(plan))
    est.parts["context"] = CONTEXT_BYTES
    est.parts["workspace"] = WORKSPACE_BYTES
    wb = weight_bytes(plan, cb, training=False, stored=work.stored_precision)
    blocks = arch.depth * block_params(arch)
    rest = total_params(arch) - blocks
    ring_params = min(max(plan.stream_resident, 2), arch.depth) * block_params(arch)
    placed = rest + (ring_params if plan.stream != "off" else blocks)
    est.parts["weights"] = int(placed * wb)
    est.parts["activations"] = int(
        work.tokens * saved_per_token(arch, "none") * work.act_factor * cb
    )
    _transients(est, work, plan)
    est.parts["io"] = work.tokens * arch.dim * cb * 4
    return est


def estimate(work: Workload, plan: MemoryPlan) -> Estimate:
    """Peak GPU bytes of one micro-step under `plan` (per rank)."""
    arch, cb = work.arch, work.compute_bytes
    est = Estimate(fragmentation=fragmentation(plan))
    est.parts["context"] = CONTEXT_BYTES
    est.parts["workspace"] = WORKSPACE_BYTES

    wb = weight_bytes(
        plan, cb, training=True, full_ft=work.full_ft, stored=work.stored_precision
    )
    # full FT: device parameters are the masters (+ the fp8 copies of fp8 compute)
    pb = work.master_bytes + wb - cb if work.full_ft else wb
    blocks = arch.depth * block_params(arch)
    rest = total_params(arch) - blocks
    shard = work.world if work.sharded else 1
    if plan.stream != "off":
        ring_params = max(plan.stream_resident, 2) * block_params(arch)
        est.parts["weights"] = int(rest * pb + ring_params * wb)
    else:
        est.parts["weights"] = int((rest + blocks) * pb) // shard
        if work.sharded and work.gathered:
            est.parts["gathered"] = int(blocks * pb)

    grad_offload = plan.grad_offload
    if work.full_ft:
        n = total_params(arch)
        grad_n = n // shard if work.sharded_grads else n
        if plan.fused_step or grad_offload or plan.stream == "swap":
            grad_n = block_params(arch)
        # a swap block's gradient is in its streaming dtype; a device master's in its own
        grad_b = cb if plan.stream == "swap" else work.master_bytes
        est.parts["grads"] = grad_n * grad_b
        if plan.optim_state == "gpu":
            est.parts["optim"] = n * state_bytes(work, plan) // shard
        swap = plan.stream == "swap"
        if swap and plan.optim_state != "cpu":
            device_states = plan.optim_state == "gpu"
            est.parts["optim_slots"] = slot_bytes(block_params(arch), cb, device_states)
        largest = largest_device_param(arch, swap)
        if plan.optim_state == "stream":
            est.parts["state_ring"] = state_ring_bytes(largest)
        if grad_offload:
            est.parts["grad_ring"] = grad_ring_bytes(largest)
        on_device = n - blocks if swap else n
        est.parts["ema"] = 4 * work.ema_decays * on_device // shard
    if work.trainable_params:
        n = work.trainable_params
        grad_b = 0 if grad_offload else 4
        est.parts["adapter"] = n * (4 + grad_b + state_bytes(work, plan)) // shard
        est.parts["ema"] = 4 * work.ema_decays * n // shard

    f = work.act_factor
    mode, frac = checkpoint_mode(plan.checkpoint)
    n_ckpt = round(arch.depth * frac)
    per_tok_full = saved_per_token(arch, "none") * f
    per_tok = saved_per_token(arch, mode) if mode != "none" else per_tok_full
    if mode == "selective":
        per_tok *= f
    stored = n_ckpt * per_tok + (arch.depth - n_ckpt) * per_tok_full
    if plan.act_offload:
        act_blocks = max(plan.act_offload_inflight, 2)
        stored = act_blocks * max(
            per_tok, per_tok_full if n_ckpt < arch.depth else per_tok
        )
    est.parts["activations"] = int(work.tokens * stored * cb)
    est.parts["recompute"] = int(work.tokens * recompute_window(arch, mode) * f * cb)
    est.parts["text_refiner"] = int(
        work.txt_tokens * arch.text_blocks * per_tok_full * cb
    )
    _transients(est, work, plan)
    est.parts["io"] = work.tokens * arch.dim * cb * 4 + work.img_tokens * (
        arch.latent_channels * 4 * 4
    )
    return est
