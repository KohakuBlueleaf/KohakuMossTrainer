"""`kohakumoss-tune`: tune the MOSS block ops and attention once for one card, dtype and
list of image sizes, and store the decisions in the per-card tuning table, so later runs
(TUNE "table" or "off", inference included) bind them without timing anything. No
weights are loaded: the op shapes are recorded from a weightless backbone
(`build_shapes`) and replayed one op at a time on fresh tensors.

    kohakumoss-tune --device cuda:0 --dtype bf16 --sizes 1024x1024 832x1216 --batch 4
    kohakumoss-tune --dtype fp16 --sizes 1024x1024 --backward   (adapter training shapes)
    kohakumoss-tune --dtype bf16 --sizes 1024x1024 --backward --trainable full

See docs/guides/logging.md and docs/internals/tuning.md.
"""

import argparse
from collections.abc import Callable

import torch

from kohakumoss.logging.log import get_logger, hl, setup, timed
from kohakumoss.model import resolve_config
from kohakumoss.model.build import DEFAULT_KEEP_FP32, build_shapes
from kohakumoss.precision import autocast_factory, autocast_for, build_precision
from kohakumoss.tuning.prewarm import ModelPrewarm, release_workspaces
from kohakumoss_data.resolution.geometry import LATENT_FACTOR
from kohakumoss_kernels.dtypes import BY_NAME
from kohakumoss_kernels.hardware.device import cache_dir
from kohakumoss_kernels.tuning import decisions, tuner

DTYPES = {name: BY_NAME[name] for name in ("bf16", "fp16")}
log = get_logger(__name__)


def record_autocast(dtype: str, device: torch.device, backward: bool) -> Callable:
    """The context factory the recorded pass runs under: the trainer's autocast for
    training shapes (`backward`), the sampler's (`autocast_for` the compute dtype, as
    `inference.pipeline.sampling_autocast`) for inference shapes."""
    if not backward:
        return autocast_for(DTYPES[dtype], device)
    return autocast_factory(build_precision(dtype, DEFAULT_KEEP_FP32), device)


def lengths(sizes: list[tuple[int, int]], text: int, batch: int, cfg: bool) -> tuple:
    """(longest joint sample, largest packed forward, shortest joint sample, samples
    per forward) in tokens for the sizes: `batch` images, doubled for CFG."""
    joint = [(h // LATENT_FACTOR) * (w // LATENT_FACTOR) + text for h, w in sizes]
    segments = batch * (2 if cfg else 1)
    return max(joint), max(joint) * segments, min(joint), segments


def tune(
    device: str,
    dtype: str,
    sizes: list[tuple[int, int]],
    variant: str = "base",
    text_tokens: int = 512,
    batch: int = 4,
    backward: bool = False,
    kernels: dict | None = None,
    trainable: str = "adapter",
) -> int:
    """Prewarm the ops of `variant` on `device` for the sizes (the record / replay
    prewarm on a weightless backbone; problems already decided are not run); returns
    problems run. With `backward`, `trainable` is "adapter" (frozen base weights,
    activations carry gradients) or "full" (every weight trains). The tuning table
    under the kernel cache receives every decision."""
    dev = torch.device(device)
    dt = DTYPES[dtype]
    full = backward and trainable == "full"
    backbone = build_shapes(resolve_config(variant), dev, dt, full, kernels=kernels)
    sample, packed, shortest, segments = lengths(
        sizes, text_tokens, batch, cfg=not backward
    )
    tuner.set_mode("table")
    loss = backbone.ops.token_mse if backward else None
    autocast = record_autocast(dtype, dev, backward)
    warm = ModelPrewarm(
        backbone, loss, autocast, dt, sample, packed, backward=backward,
        segments=segments, shortest=shortest, input_grad=not full,
    )  # fmt: skip
    what = f"tuning {variant} {dtype} on {dev} (sample {sample}, batch {packed} tokens)"
    with timed(log, what) as done:
        measured = warm([dev])
        done["problems"] = measured
    release_workspaces([dev])
    decisions.flush("kohakumoss-tune")
    log.info(f"tuning table: {hl(cache_dir() / 'tuning', 'path')}")
    return measured


def _size(text: str) -> tuple[int, int]:
    h, w = text.lower().split("x")
    return int(h), int(w)


def main(argv: list[str] | None = None) -> None:
    setup()
    ap = argparse.ArgumentParser(prog="kohakumoss-tune", description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    ap.add_argument("--sizes", nargs="+", type=_size, default=[(1024, 1024)])
    ap.add_argument("--variant", default="base")
    ap.add_argument("--text-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4, help="images per generate batch")
    ap.add_argument("--backward", action="store_true", help="training (fwd + bwd)")
    ap.add_argument(
        "--trainable",
        choices=("adapter", "full"),
        default="adapter",
        help="with --backward: frozen base (adapters) or every weight trains",
    )
    args = ap.parse_args(argv)
    tune(
        args.device,
        args.dtype,
        args.sizes,
        args.variant,
        args.text_tokens,
        args.batch,
        args.backward,
        trainable=args.trainable,
    )


if __name__ == "__main__":
    main()
