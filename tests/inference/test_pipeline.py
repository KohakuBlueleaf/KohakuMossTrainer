"""Packed CFG equals two-pass CFG; per-sample determinism; pipeline plumbing."""

import json

import pytest
import torch

from kohakumoss.diffusion.sampling import sample_latents
from kohakumoss.inference import MossPipeline
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail
from tests.inference.conftest import FakeTextEncoder, FakeVae

pytestmark = pytest.mark.cpu


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _embeds(dim, lens, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, dim, generator=g, dtype=torch.float64) for n in lens]


def test_segment_skip_equals_skipping(tiny64):
    cfg = tiny64.cfg
    hw = torch.tensor([[3, 4], [2, 5]])
    lens = torch.tensor([6, 3])
    seq = build_seqinfo(hw, lens)
    g = torch.Generator().manual_seed(0)
    img = torch.randn(22, cfg.latent_channels, generator=g, dtype=torch.float64)
    text = torch.randn(9, cfg.text_dim, generator=g, dtype=torch.float64)
    sigma = torch.tensor([0.7, 0.3])
    skip = frozenset(cfg.negative_skip_layers)
    mask = torch.tensor([False, True])
    seg = tiny64.bind_forward(segment_skip_layers=tuple(skip))(
        img, text, sigma, seq, mask
    )
    full = tiny64(img, text, sigma, seq)
    skipped = tiny64.bind_forward(skip)(img, text, sigma, seq)
    assert torch.equal(seg[:12], full[:12])
    assert _rel(seg[12:], skipped[12:]) < 1e-12
    must_fail(lambda: _assert(_rel(full[12:], skipped[12:]) < 1e-12), "skip has effect")


def _assert(cond):
    assert cond


def test_packed_cfg_equals_two_pass(tiny64):
    cfg = tiny64.cfg
    pos = _embeds(cfg.text_dim, (5, 8, 2), 1)
    neg = _embeds(cfg.text_dim, (2, 2, 2), 2)
    kw = dict(steps=5, cfg=3.0, seeds=(3, 4, 5), device="cpu")
    sizes = [(48, 64), (32, 32), (64, 16)]
    packed = sample_latents(tiny64, pos, neg, sizes, cfg_mode="packed", **kw)
    twopass = sample_latents(tiny64, pos, neg, sizes, cfg_mode="two_pass", **kw)
    assert _rel(packed.latents, twopass.latents) < 1e-6
    noskip = sample_latents(tiny64, pos, neg, sizes, skip_layers=(), **kw)
    must_fail(lambda: _assert(_rel(noskip.latents, twopass.latents) < 1e-6), "skip")


def test_sample_independent_of_batch(tiny64, fake_te):
    pipe = MossPipeline.from_components(tiny64, fake_te, FakeVae(), device="cpu")
    prompts = ["a red apple", "two cats on a sofa at night", "x"]
    kw = dict(
        sizes=[(32, 48), (64, 32), (32, 32)], steps=4, seeds=[7, 8, 9], output="latent"
    )
    together = pipe.generate(prompts, batch=3, **kw).latents
    alone = pipe.generate(prompts, batch=1, **kw).latents
    for a, b in zip(together, alone):
        assert a.shape == b.shape and _rel(a, b) < 1e-6


def test_generate_outputs_and_settings(tiny64, fake_te):
    pipe = MossPipeline.from_components(tiny64, fake_te, FakeVae(), device="cpu")
    res = pipe.generate(
        ["a photo of a dog"],
        negative_prompts=["blurry"],
        sizes=(32, 48),
        steps=3,
        seeds=[1],
    )
    assert res.images[0].size == (48, 32)
    json.dumps(res.settings)
    assert res.settings["skip_layers"] == list(tiny64.cfg.negative_skip_layers)
    tiny64.train()
    pipe.generate(["a"], sizes=(32, 32), steps=1, output="latent")
    assert tiny64.training


def test_negative_strings_with_prompt_embeds(tiny64, fake_te):
    pipe = MossPipeline.from_components(tiny64, fake_te, FakeVae(), device="cpu")
    pos = fake_te.encode(["a photo of a dog", "x"])
    negs = ["blurry, low quality", ""]
    kw = dict(prompt_embeds=pos, sizes=(32, 32), steps=2, seeds=[1, 2])
    kw.update(output="latent")
    strings = pipe.generate(negative_prompts=negs, **kw).latents
    neg = [fake_te.encode([negs[0]])[0], fake_te.empty()]
    explicit = pipe.generate(negative_embeds=neg, **kw).latents
    empty = pipe.generate(**kw).latents
    assert all(torch.equal(a, b) for a, b in zip(strings, explicit))
    must_fail(
        lambda: _assert(all(torch.equal(a, b) for a, b in zip(empty, explicit))),
        "negative strings dropped",
    )


def test_text_once_frees_encoder(tiny64):
    te = FakeTextEncoder(64)
    pipe = MossPipeline(tiny64, te, FakeVae(), device="cpu", residency="te_once")
    pipe.generate(["a", "b"], sizes=(32, 32), steps=1, output="latent")
    assert pipe.place.text_encoder is None
    with pytest.raises(RuntimeError):
        pipe.generate(["c"], sizes=(32, 32), steps=1, output="latent")
    pipe.generate(
        prompt_embeds=te.encode(["c"]), sizes=(32, 32), steps=1, output="latent"
    )
