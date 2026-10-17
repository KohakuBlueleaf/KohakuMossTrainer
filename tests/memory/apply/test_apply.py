"""MemoryPlans applied to the real (tiny) MossBackbone, for training and inference."""

import pytest
import torch

from kohakumoss.model import random_backbone
from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.training.build.model import build_model
from kohakumoss.training.knobs import resolve_precision
from kohakumoss.training.memory.apply import apply_inference_plan, apply_to_backbone
from kohakumoss.training.memory.profiles import resolve
from kohakumoss.training.precision import build_policy
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail
from tests.training.test_entry_smoke import smoke_knobs

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


def _inference(overrides, corrupt=None):
    model = _model()
    plan = resolve("minimal", overrides, device="cpu", inference=True)
    engines = apply_inference_plan(plan, "cpu", torch.float32, backbone=model)
    if corrupt is not None:
        corrupt(engines)
    model.bind_ops()
    return model(*_inputs(model)), engines


def _swap_blocks(engines) -> None:
    host = engines.stream.host
    host[0], host[1] = host[1], host[0]


@torch.no_grad()
def test_inference_minimal_plan_equals_resident_int8():
    ref, _ = _inference({"stream": "off"})
    out, engines = _inference(None)
    assert engines.stream is not None and engines.reports["quant"]["layers"] > 0

    def check(got):
        assert ((got - ref).norm() / ref.norm()).item() < 1e-6

    check(out)
    wrong, _ = _inference(None, corrupt=_swap_blocks)
    must_fail(lambda: check(wrong), "a stream that loads the neighbouring block")


def test_trainer_build_excludes_adapters_from_activation_offload(tmp_path):
    """Adapters bound after the plan are never offloaded (their storages are skipped)."""
    knobs = smoke_knobs(tmp_path, MEMORY_OVERRIDES={"act_offload": True})
    policy = build_policy(resolve_precision(knobs), knobs["KEEP_FP32"])
    built = build_model(knobs, torch.device("cpu"), policy)
    off = built.engines.offload
    params = list(built.trainable.trainable_params())
    assert params

    def check(engine):
        for p in params:
            assert engine.pack(p) is p, "trainable parameter would be offloaded"

    off.min_bytes = 0
    check(off)
    fresh = ActivationOffload("cpu", min_bytes=0)
    must_fail(lambda: check(fresh), "offload without the adapter exclusion")


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
