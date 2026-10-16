"""Analytic GPU-memory model of one training step: weights, trainable state, activations.

The constants are per-token saved-activation counts of the MOSS joint block; see
docs/internals/offload.md for their derivation and the probe that calibrates them.
"""

from dataclasses import dataclass, field

from kohakumoss.training.memory.profiles import checkpoint_mode
from kohakumoss.types.arch import MossArchConfig
from kohakumoss.types.plans import MemoryPlan

GIB = 1024**3
PRECISION_BYTES = {
    "compute": None,
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp8": 1,
    "int8": 1,
}
OPTIM_BYTES = {"gpu": 8, "stream": 0, "8bit": 2, "cpu": 0}
CONTEXT_BYTES = int(0.45 * GIB)
WORKSPACE_BYTES = int(0.25 * GIB)
FRAGMENTATION = 1.08
ACT_FACTOR = {"fused": 1.0, "eager": 3.6}


@dataclass(frozen=True)
class Workload:
    """What one rank runs per micro-step."""

    arch: MossArchConfig
    img_tokens: int
    txt_tokens: int
    trainable_params: int = 0
    full_ft: bool = False
    compute_bytes: int = 2
    world: int = 1
    sharded: bool = False
    act_factor: float = ACT_FACTOR["fused"]

    @property
    def tokens(self) -> int:
        return self.img_tokens + self.txt_tokens


@dataclass
class Estimate:
    """Bytes per component and their total (with context, workspace, fragmentation)."""

    parts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(sum(self.parts.values()) * FRAGMENTATION)

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


def estimate(work: Workload, plan: MemoryPlan) -> Estimate:
    """Peak GPU bytes of one micro-step under `plan` (per rank)."""
    arch, cb = work.arch, work.compute_bytes
    est = Estimate()
    est.parts["context"] = CONTEXT_BYTES
    est.parts["workspace"] = WORKSPACE_BYTES

    wb = PRECISION_BYTES.get(plan.base_precision) or cb
    blocks = arch.depth * block_params(arch)
    rest = total_params(arch) - blocks
    shard = work.world if work.sharded else 1
    if plan.stream != "off":
        resident = max(plan.stream_resident, 2) * block_params(arch)
        est.parts["weights"] = (rest + resident) * wb
    else:
        est.parts["weights"] = (rest + blocks) * wb // shard

    grad_offload = plan.grad_offload
    if work.full_ft:
        n = total_params(arch)
        grad_n = n // shard
        if plan.fused_step or grad_offload or plan.stream == "swap":
            grad_n = block_params(arch)
        est.parts["grads"] = grad_n * cb
        if plan.optim_state in ("gpu", "8bit"):
            est.parts["master"] = n * 4 // shard
            est.parts["optim"] = n * OPTIM_BYTES[plan.optim_state] // shard
        if plan.optim_state == "stream":
            est.parts["optim_slots"] = 2 * block_params(arch) * 16
    if work.trainable_params:
        n = work.trainable_params
        grad_b = 0 if grad_offload else 4
        est.parts["adapter"] = n * (4 + grad_b + OPTIM_BYTES[plan.optim_state]) // shard

    f = work.act_factor
    mode, frac = checkpoint_mode(plan.checkpoint)
    n_ckpt = round(arch.depth * frac)
    per_tok_full = saved_per_token(arch, "none") * f
    per_tok = saved_per_token(arch, mode) if mode != "none" else per_tok_full
    if mode == "selective":
        per_tok *= f
    stored = n_ckpt * per_tok + (arch.depth - n_ckpt) * per_tok_full
    if plan.act_offload:
        stored = 2 * max(per_tok, per_tok_full if n_ckpt < arch.depth else per_tok)
    est.parts["activations"] = int(work.tokens * stored * cb)
    est.parts["recompute"] = int(work.tokens * recompute_window(arch, mode) * f * cb)
    est.parts["text_refiner"] = int(
        work.txt_tokens * arch.text_blocks * per_tok_full * cb
    )
    if plan.base_precision in ("fp8", "int8"):
        largest = 2 * arch.ffn_dim * arch.dim
        est.parts["dequant"] = (block_params(arch) + 2 * largest) * cb
    est.parts["io"] = work.tokens * arch.dim * cb * 4 + work.img_tokens * (
        arch.latent_channels * 4 * 4
    )
    return est
