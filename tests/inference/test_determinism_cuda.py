"""Batch composition changes a sample only up to kernel rounding on CUDA (bf16)."""

import pytest
import torch

from kohakumoss.inference import MossPipeline
from kohakumoss.model import build_backbone, resolve_config
from kohakumoss.model.opset import FUSED, PRIMITIVES
from tests.inference.conftest import FakeTextEncoder

pytestmark = pytest.mark.cuda


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.mark.parametrize("kernels", ["eager", "arch"])
def test_batch_composition_rounding_only(kernels):
    choice = {n: "eager" for n in PRIMITIVES + FUSED} if kernels == "eager" else None
    cfg = resolve_config("tiny")
    model = build_backbone(
        cfg, device="cuda", dtype=torch.bfloat16, init=True, seed=2, kernels=choice
    )
    te = FakeTextEncoder(model.cfg.text_dim)
    pipe = MossPipeline(model, te, None, device="cuda", owned=True, tune="off")
    prompts = ["a red apple", "two cats on a sofa at night", "x"]
    kw = dict(
        sizes=[(32, 48), (64, 32), (32, 32)], steps=4, seeds=[7, 8, 9], output="latent"
    )
    together = pipe.generate(prompts, batch=3, **kw).latents
    alone = pipe.generate(prompts, batch=1, **kw).latents
    errs = [_rel(a, b) for a, b in zip(together, alone)]
    print(kernels, "batch-composition rel errors", errs)
    assert max(errs) < 3e-2, errs
    other = pipe.generate(prompts, batch=1, **{**kw, "seeds": [1, 2, 3]}).latents
    assert min(_rel(a, b) for a, b in zip(other, alone)) > 10 * max(errs + [1e-3])
