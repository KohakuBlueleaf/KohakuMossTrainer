"""MemoryPlans applied to the real (tiny) MossBackbone, for training and inference."""

import pytest
import torch

from kohakumoss.model import random_backbone
from kohakumoss.training.memory.apply import apply_inference_plan, apply_to_backbone
from kohakumoss.training.memory.profiles import resolve
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SIZE = dict(dim=256, heads=4, ffn_dim=512)


def _inputs(model, seed=0):
    cfg = model.cfg
    g = torch.Generator().manual_seed(seed)
    seq = build_seqinfo(torch.tensor([[4, 6], [3, 3]]), torch.tensor([5, 7]))
    img = torch.randn(seq.n_img, cfg.latent_channels, generator=g)
    text = torch.randn(seq.n_txt, cfg.text_dim, generator=g)
    sigma = torch.tensor([0.3, 0.8])
    return img, text, sigma, seq


def _model():
    return random_backbone("tiny", dtype=torch.float32, **SIZE)


@torch.no_grad()
def test_inference_minimal_plan_close_to_dense():
    ref_model = _model()
    ref = ref_model(*_inputs(ref_model))
    model = _model()
    plan = resolve("minimal", device="cpu", inference=True)
    engines = apply_inference_plan(plan, "cpu", torch.float32, backbone=model)
    model.bind_ops()
    out = model(*_inputs(model))
    assert engines.stream is not None and engines.reports["quant"]["layers"] > 0
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 5e-2

    def check_unquantized_equality():
        assert rel < 1e-6

    must_fail(check_unquantized_equality, "int8 base is not bit-equal to dense")


def _train_grads(plan_overrides):
    model = _model()
    for p in model.parameters():
        p.requires_grad_(False)
    model.video_embedder.weight.requires_grad_(True)
    model.time_embedding[0].weight.requires_grad_(True)
    if plan_overrides is not None:
        plan = resolve("balanced", plan_overrides, device="cpu")
        apply_to_backbone(model, plan, "cpu", torch.float32, training=True)
        model.bind_ops()
    v = model(*_inputs(model))
    (v.square().mean()).backward()
    return (
        model.video_embedder.weight.grad.clone(),
        model.time_embedding[0].weight.grad.clone(),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"checkpoint": "block"},
        {"checkpoint": "selective"},
        {"checkpoint": "block", "stream": "ring", "stream_resident": 2},
        {"checkpoint": "block", "act_offload": True},
    ],
)
def test_training_plans_keep_gradients(overrides):
    ref = _train_grads(None)
    got = _train_grads(overrides)
    for a, b in zip(ref, got):
        assert (a - b).abs().max().item() < 1e-5 * (a.abs().max().item() + 1e-12)
