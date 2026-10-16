"""Exported LoRA files loaded by MossPipeline reproduce the live adapted model."""

import pytest
import torch

from kohakumoss.adapters import AdapterSpec, apply
from kohakumoss.inference import MossPipeline
from kohakumoss.model import load_backbone, random_backbone, save_single
from tests._util.negative import must_fail
from tests.inference.conftest import FakeTextEncoder
from tests.model.conftest import make_batch

pytestmark = pytest.mark.cpu
MULT = 0.7


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lora")
    save_single(random_backbone("tiny", seed=21), tmp / "base.safetensors", dtype=None)
    live, _ = load_backbone(tmp / "base.safetensors", device="cpu", dtype=torch.float32)
    spec = AdapterSpec(algo="lora", preset="attn-mlp", dim=4, alpha=2.0)
    aset = apply(live, spec)
    torch.manual_seed(5)
    with torch.no_grad():
        for p in aset.trainable_params():
            p.normal_(0, 0.5)
    aset.set_multiplier(MULT)
    live.bind_ops()
    for fmt in ("comfy", "lycoris"):
        aset.export(str(tmp / f"{fmt}.safetensors"), fmt, dtype=None)
    return tmp, live


def _pipe(tmp, fmt, mode, mult):
    return MossPipeline.from_pretrained(
        tmp,
        transformer=tmp / "base.safetensors",
        dtype=torch.float32,
        device="cpu",
        adapters=[(str(tmp / f"{fmt}.safetensors"), mult)],
        adapter_mode=mode,
        text_encoder=False,
        vae=False,
    )


@pytest.mark.parametrize("fmt", ["comfy", "lycoris"])
@pytest.mark.parametrize("mode", ["merge", "bypass"])
def test_loaded_adapter_matches_live(files, fmt, mode):
    tmp, live = files
    img, text, sigma, seq, _, _ = make_batch(live.cfg)
    with torch.no_grad():
        ref = live(img, text, sigma, seq)
        base = load_backbone(
            tmp / "base.safetensors", device="cpu", dtype=torch.float32
        )
        plain = base[0](img, text, sigma, seq)
        pipe = _pipe(tmp, fmt, mode, MULT)
        out = pipe.backbone(img, text, sigma, seq)
    assert _rel(plain, ref) > 1e-3
    assert _rel(out, ref) < 1e-5, _rel(out, ref)
    wrong = _pipe(tmp, fmt, mode, 1.0)
    with torch.no_grad():
        off = wrong.backbone(img, text, sigma, seq)
    must_fail(lambda: _check(_rel(off, ref) < 1e-5), "multiplier ignored")

    te = FakeTextEncoder(live.cfg.text_dim)
    kw = dict(sizes=(32, 48), steps=3, seeds=[4], output="latent")
    prompt = te.encode(["a b c"])
    a = MossPipeline.from_components(live, device="cpu").generate(
        prompt_embeds=prompt, negative_embeds=te.encode([""]), **kw
    )
    b = pipe.generate(prompt_embeds=prompt, negative_embeds=te.encode([""]), **kw)
    assert _rel(b.latents[0], a.latents[0]) < 1e-5


def _check(cond):
    assert cond
