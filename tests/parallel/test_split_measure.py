"""The measured stage split, model-free (the MOSS-Image base architecture, all 42 blocks,
on the meta device; no weight is created or read; every parameter trainable as under
full finetune): a block's parameter gradients are gone once the next block is timed, so
at most one block's gradients exist at a time, and none is left after the
measurement."""

import contextlib

import pytest
import torch

from kohakumoss.memory.runners import PlainRunner
from kohakumoss.model import build_backbone, resolve_config
from kohakumoss.model.opset import OpSet
from kohakumoss.parallel.pipeline.distributed import PipeShape
from kohakumoss.parallel.pipeline.parts import BackboneParts
from kohakumoss.parallel.pipeline.split import measure_costs
from kohakumoss.precision import build_precision
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = [pytest.mark.cpu, pytest.mark.budget(30)]
META = torch.device("meta")


class HeldGrads:
    """Runner that records, at every block call, the other blocks holding gradients."""

    def __init__(self, layers):
        self.layers = layers
        self.inner = PlainRunner().bind(layers)
        self.held: list[list[int]] = []

    def run(self, index, block, x, ctx):
        self.held.append(
            [
                j
                for j, b in enumerate(self.layers)
                if j != index and any(p.grad is not None for p in b.parameters())
            ]
        )
        return self.inner.run(index, block, x, ctx)


def _parts() -> BackboneParts:
    k = runnable()
    prec = build_precision("bf16", tuple(k["KEEP_FP32"]))
    backbone = build_backbone(
        resolve_config(k["VARIANT"]), META, prec.compute, prec.keep_fp32
    )
    backbone.bind_ops(OpSet.resolve(torch.device("cpu"), prec.compute, {}))
    # RoPE bands to the meta device, next to the block inputs
    backbone.rope.bands = backbone.rope.bands.to(META)
    backbone.requires_grad_(True)
    return BackboneParts(backbone)


def check_one_block_of_grads(parts: BackboneParts) -> None:
    shape = PipeShape(
        tokens=1024, max_samples=2, dim=parts.cfg.dim, microbatches=1, sample_tokens=512
    )
    runner = HeldGrads(parts.layers)
    measure_costs(parts, shape, META, contextlib.nullcontext, runner, warmup=1, iters=1)
    assert len(runner.held) >= len(parts.layers), "not every block was timed"
    assert all(not held for held in runner.held), runner.held
    assert all(p.grad is None for p in parts.parameters())


def test_measured_split_holds_one_block_of_grads():
    parts = _parts()
    assert len(parts.layers) == 42
    check_one_block_of_grads(parts)
    for block in parts.layers:
        block.zero_grad = lambda set_to_none=True: None
    must_fail(
        lambda: check_one_block_of_grads(parts), "every block keeps its gradients"
    )
