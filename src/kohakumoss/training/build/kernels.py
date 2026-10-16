"""Kernel prewarm before step 0 and the between-step retune of uncovered problems.

Prewarm runs the backbone's resolved attention and block ops forward + backward on
scratch inputs over every length bucket the planned batches reach, outside any
checkpoint / no_grad / inference mode, so live calls only look tuned choices up. Both
actions are bound at build time; TUNE="off" binds neither (nothing is timed).
"""

import math

import torch
import torch.distributed as dist

from kohakumoss.registry import import_object
from kohakumoss.training.build import integration as I

BLOCK_OPS = ("rms_norm_mod", "rms_norm", "gated_residual", "situ_glu", "qk_norm_rope")
BLOCK_FLOOR = 256  # smallest token bucket of the block-op problem keys


def _ceil_to(x: float, step: int) -> int:
    return int(math.ceil(x / step)) * step


def image_tokens_max(cfg) -> int:
    """Largest latent grid (tokens) the resolution planner yields for `cfg`: the area
    target at the aspect extremes and 1, each side rounded up to the size step."""
    best = 0
    for aspect in (*cfg.aspect_range, 1.0):
        w = _ceil_to(math.sqrt(cfg.target_area * aspect), cfg.size_step)
        h = _ceil_to(math.sqrt(cfg.target_area / aspect), cfg.size_step)
        best = max(best, (w // cfg.latent_factor) * (h // cfg.latent_factor))
    return best


def planned_lengths(cfg) -> tuple[int, int]:
    """(longest joint sample, largest joint batch) in tokens for data config `cfg`."""
    sample = min(cfg.max_tokens, image_tokens_max(cfg) + cfg.max_text_tokens)
    batch = cfg.max_tokens if cfg.batching.startswith("tokens") else cfg.batch_size * sample
    return sample, batch


def block_lengths(batch_max: int) -> list[list[int]]:
    """One single-sample batch per power-of-two token bucket up to `batch_max`."""
    out, n = [], BLOCK_FLOOR
    while True:
        out.append([n])
        if n >= batch_max:
            return out
        n *= 2


def _noop(*args) -> None:
    return None


class Prewarm:
    """Tune attention + block ops of `ops` on each device for the planned lengths."""

    def __init__(self, ops, arch, dtype: torch.dtype, sample_max: int, batch_max: int):
        self.ops = ops
        self.arch = arch
        self.dtype = dtype
        self.sample_max = sample_max
        self.lengths = block_lengths(batch_max)
        self.attention = import_object(I.KERNEL_PREWARM_ATTENTION)
        self.blocks = import_object(I.KERNEL_PREWARM_BLOCKS)
        self.broadcast = import_object(I.KERNEL_TUNE_BROADCAST)

    def __call__(self, devices: list[torch.device]) -> int:
        a = self.arch
        block_ops = {name: getattr(self.ops, name) for name in BLOCK_OPS}
        measured = 0
        for device in devices:
            self.attention(
                self.ops.attention_varlen,
                device,
                self.dtype,
                a.heads,
                a.head_dim,
                self.sample_max,
            )
            measured += self.blocks(
                block_ops, device, self.dtype, self.lengths, a.dim, a.heads, a.ffn_dim
            )
        if dist.is_available() and dist.is_initialized():
            self.broadcast(0)
        return measured


def build_prewarm(k: dict, backbone, dtype: torch.dtype):
    """The prewarm action of the run (a no-op under TUNE='off')."""
    if k["TUNE"] == "off":
        return _noop
    cfg = import_object(I.DATA_CONFIG).from_knobs(k)
    sample, batch = planned_lengths(cfg)
    return Prewarm(backbone.ops, backbone.cfg, dtype, sample, batch)


def prewarm_devices(k: dict, device: torch.device) -> list[torch.device]:
    """Devices whose kernels this process runs (every stage device when threaded);
    none on CPU, where every op is the untuned eager spec."""
    if device.type == "cpu":
        return []
    if k["PARALLEL"] == "threaded":
        return [torch.device(device.type, i) for i in k["DEVICES"]]
    return [device]


class RetuneKernels:
    """Every step: decide raced-op problems first met in this step (outside autograd)."""

    every = 1

    def __init__(self):
        self.retune = import_object(I.KERNEL_RETUNE)

    def on_step_end(self, state, out) -> None:
        self.retune()


def retune_callbacks(k: dict) -> list:
    return [] if k["TUNE"] == "off" else [RetuneKernels()]
