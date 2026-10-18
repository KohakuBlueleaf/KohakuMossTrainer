"""Sampling on the real pipeline (DiT under the inference `balanced` plan, text encoder
streamed, VAE offloaded, tuning off) with two of the dataset's captions: packed CFG and
two-pass CFG give the same latents (within a few times the difference of two-pass CFG
run one sample at a time); the sampler's autocast (16-bit inputs to the conditioning
linears) moves the samples by no more than that same batch-composition noise, against
fp32-input conditioning, in bf16 and in fp16 (the DiT from the fp16-fix file); and a generate builds no autograd state (no module
runs with grad enabled, no tensor is saved for backward, no output carries a graph)."""

import contextlib

import pytest
import torch

from kohakumoss.diffusion.sampling import sample as sample_module
from kohakumoss.inference import MossPipeline
from kohakumoss.inference import pipeline as pipeline_module
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
NEGATIVE = ["lowres, blurry, bad anatomy", ""]
GEN = dict(
    sizes=[(256, 256), (192, 320)],
    steps=4,
    cfg=4.0,
    seeds=[3, 9],
    batch=2,
    progress=False,
)
CFG_GEN = dict(GEN, steps=1)
NOISE = 3.0  # see docs/internals/fp16.md#real-model-structural-tests


@pytest.fixture(scope="module")
def pipe(weights_root):
    return MossPipeline.from_pretrained(
        weights_root / "base",
        dtype=torch.bfloat16,
        device="cuda",
        vae_plan={"memory": "offload"},
        text_plan={"memory": "stream"},
        memory="balanced",
        tune="off",
    )


def _rel(a, b) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def noise_floor(got: list, want: list, dtype=torch.bfloat16) -> list[float]:
    """Per sample: the relative difference of two runs of the same math, at least the
    rounding scale of `dtype`."""
    eps = torch.finfo(dtype).eps
    return [max(_rel(a, b), eps) for a, b in zip(got, want)]


def check_same(got: list, want: list, floor: list[float]) -> None:
    """Per sample: finite, same shape, within NOISE x the sample's noise floor."""
    assert len(got) == len(want) == len(floor)
    for a, b, f in zip(got, want, floor):
        assert torch.isfinite(a).all() and a.shape == b.shape, a.shape
        assert _rel(a, b) <= NOISE * f, (_rel(a, b), f)


def _latents(pipe, captions, cfg_mode: str, gen: dict = GEN) -> list:
    res = pipe.generate(captions, NEGATIVE, cfg_mode=cfg_mode, output="latent", **gen)
    return [z.cpu() for z in res.latents]


def _uncond_reads_positive(backbone, hw, positive, negative, cfg, skip, device):
    return sample_module._packed(backbone, hw, positive, positive, cfg, skip, device)


def test_packed_cfg_equals_two_pass(pipe, captions, monkeypatch):
    two_pass = _latents(pipe, captions, "two_pass", CFG_GEN)
    alone = _latents(pipe, captions, "two_pass", dict(CFG_GEN, batch=1))
    floor = noise_floor(alone, two_pass)
    packed = _latents(pipe, captions, "packed", CFG_GEN)
    check_same(packed, two_pass, floor)
    monkeypatch.setitem(sample_module.CFG_MODES, "packed", _uncond_reads_positive)
    broken = _latents(pipe, captions, "packed", CFG_GEN)
    must_fail(
        lambda: check_same(broken, two_pass, floor), "packed CFG fed the positive text"
    )


def _fp32_conditioning(backbone, device):
    return contextlib.nullcontext


def _fp16_pipe(weights_root, fix_file):
    """The fp16 pipeline: the DiT from the fp16-fix file (its fold carried, fp16_fix
    "none") under `low`, so it fits next to the module's bf16 pipeline."""
    return MossPipeline.from_pretrained(
        weights_root / "base",
        transformer=fix_file,
        dtype=torch.float16,
        device="cuda",
        vae_plan={"memory": "offload"},
        text_plan={"memory": "stream"},
        memory="low",
        tune="off",
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_sampling_autocast_moves_samples_like_batch_noise(
    dtype, weights_root, fix_file, captions, monkeypatch, request
):
    if dtype == torch.float16:
        if fix_file is None:
            pytest.skip("no fp16-fix file")
        p = _fp16_pipe(weights_root, fix_file)
    else:
        p = request.getfixturevalue("pipe")
    for steps in (1, 4):
        gen = dict(GEN, steps=steps)
        shipped = _latents(p, captions, "two_pass", gen)
        alone = _latents(p, captions, "two_pass", dict(gen, batch=1))
        floor = noise_floor(alone, shipped, dtype)
        with monkeypatch.context() as m:
            m.setattr(pipeline_module, "sampling_autocast", _fp32_conditioning)
            fp32_inputs = _latents(p, captions, "two_pass", gen)
        assert any(_rel(a, b) > 0 for a, b in zip(fp32_inputs, shipped)), steps
        check_same(fp32_inputs, shipped, floor)
        with monkeypatch.context() as m:
            m.setitem(sample_module.CFG_MODES, "packed", _uncond_reads_positive)
            broken = _latents(p, captions, "packed", gen)
        must_fail(
            lambda: check_same(broken, shipped, floor),
            f"packed CFG fed the positive text ({steps} steps)",
        )
    del p
    torch.cuda.empty_cache()


def traced(pipe, run) -> tuple[dict, object]:
    """Run `run()`; the backbone modules entered with grad enabled, the tensors saved
    for backward, and whether any output latent carries autograd state."""
    seen, saved, handles = [], [0], []
    for name, m in pipe.backbone.named_modules():
        handles.append(
            m.register_forward_pre_hook(
                lambda mod, args, name=name: (
                    seen.append(name) if torch.is_grad_enabled() else None
                )
            )
        )

    def pack(t):
        saved[0] += 1
        return t

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            res = run()
    finally:
        for h in handles:
            h.remove()
    tracked = any(z.requires_grad or z.grad_fn is not None for z in res.latents)
    return {"grad_modules": seen, "saved": saved[0], "tracked": tracked}, res


def check_untracked(trace: dict) -> None:
    assert trace["grad_modules"] == [], trace["grad_modules"][:5]
    assert trace["saved"] == 0, trace["saved"]
    assert not trace["tracked"]


def _trainable(module) -> list[str]:
    return [n for n, p in module.named_parameters() if p.requires_grad]


def test_generate_holds_no_autograd_state(pipe, captions, monkeypatch):
    assert _trainable(pipe.backbone) == []
    trace, res = traced(
        pipe, lambda: pipe.generate(captions, NEGATIVE, output="pil", **GEN)
    )
    check_untracked(trace)
    assert len(res.images) == len(captions)
    monkeypatch.setattr(pipe, "_grad_ctx", torch.enable_grad)
    one = dict(GEN, sizes=(256, 256), seeds=[0], batch=1, steps=2, cfg=1.0)
    graded, _ = traced(
        pipe, lambda: pipe.generate(captions[:1], output="latent", **one)
    )
    must_fail(lambda: check_untracked(graded), "a generate that runs with grad enabled")
    leaf = torch.ones(2, requires_grad=True)
    saving, _ = traced(pipe, lambda: _Latents([leaf.exp()]))
    must_fail(lambda: check_untracked(saving), "a run that saves a tensor for backward")


class _Latents:
    """A result holding `latents` only."""

    def __init__(self, latents):
        self.latents = latents
