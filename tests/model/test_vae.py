"""FluxVae against the official normalisation lines, on a stub VAE."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kohakumoss.encoders import vae as vae_module
from kohakumoss.encoders.vae import FluxVae, vae_encoder
from kohakumoss_data.scripting import build_encoder
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
EPS = 1e-4


class _Posterior:
    def __init__(self, z):
        self.z = z

    def mode(self):
        return self.z

    def sample(self, generator=None):
        return self.z + 1.0


class StubVae(nn.Module):
    """Posterior mode = per-channel scaled 8x8 average of the pixels; records decode input."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.bn = nn.BatchNorm2d(128)
        self.bn.running_mean.copy_(torch.randn(128, generator=g))
        self.bn.running_var.copy_(torch.rand(128, generator=g) + 0.5)
        self.config = SimpleNamespace(batch_norm_eps=EPS)
        self.scale = nn.Parameter(torch.linspace(0.5, 2.0, 32).view(1, 32, 1, 1))
        self.decoded = None

    def encode(self, x):
        base = F.avg_pool2d(x.mean(1, keepdim=True), 8)
        ramp = torch.arange(base.shape[-1], dtype=x.dtype).view(1, 1, 1, -1)
        return SimpleNamespace(latent_dist=_Posterior(base * self.scale + 0.1 * ramp))

    def decode(self, z, return_dict=False):
        self.decoded = z
        return (torch.zeros(z.shape[0], 3, 8 * z.shape[2], 8 * z.shape[3]),)


def _official_encode(stub, pixels):
    """diffusers Flux2 patchify + MOSS BatchNorm normalisation, restated."""
    z = stub.encode(pixels).latent_dist.mode()
    b, c, h, w = z.shape
    z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(b, c * 4, h // 2, w // 2)
    bn = stub.bn
    return (z - bn.running_mean.view(1, -1, 1, 1)) / (
        bn.running_var.view(1, -1, 1, 1) + EPS
    ).sqrt()


def _official_decode_input(stub, z):
    """The decode lines of MOSSImagePipeline.__call__, verbatim."""
    bn = stub.bn
    z = z * (bn.running_var + EPS).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(
        z
    ).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    return (
        z.reshape(b, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c // 4, 2 * h, 2 * w)
    )


def _check(cond):
    assert cond


def test_encode_decode_match_official():
    stub = StubVae()
    vae = FluxVae(stub)
    pixels = torch.rand(2, 3, 64, 96) * 2 - 1
    z = vae.encode(pixels)
    ref = _official_encode(stub, pixels)
    assert torch.allclose(z, ref, atol=1e-6)
    vae.decode(z)
    assert torch.allclose(stub.decoded, _official_decode_input(stub, z), atol=1e-6)
    assert torch.allclose(
        stub.decoded, stub.encode(pixels).latent_dist.mode(), atol=1e-5
    )

    broken = FluxVae(stub)
    broken.std = stub.bn.running_var.float() + EPS
    must_fail(
        lambda: _check(torch.allclose(broken.encode(pixels), ref, atol=1e-6)), "no sqrt"
    )
    swapped = FluxVae(stub)
    swapped.mean, swapped.std = vae.std.clone(), vae.mean.clone()
    must_fail(
        lambda: _check(torch.allclose(swapped.encode(pixels), ref, atol=1e-6)), "swap"
    )


def test_sample_mode_is_selected_once():
    stub = StubVae()
    pixels = torch.rand(1, 3, 32, 32) * 2 - 1
    a = FluxVae(stub, "mode").encode(pixels)
    b = FluxVae(stub, "sample").encode(pixels)
    assert not torch.allclose(a, b)


def test_encode_images_packs_row_major():
    stub = StubVae()
    vae = FluxVae(stub)
    rng = np.random.default_rng(0)
    images = [
        rng.integers(0, 255, (32, 48, 3), dtype=np.uint8),
        rng.integers(0, 255, (64, 32, 3), dtype=np.uint8),
    ]
    hw = torch.tensor([[2, 3], [4, 2]])
    packed = vae.encode_images(images, hw)
    rows = []
    for img, (h, w) in zip(images, hw.tolist()):
        px = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        z = _official_encode(stub, px)[0]
        rows += [z[:, y, x] for y in range(h) for x in range(w)]
    ref = torch.stack(rows)
    assert torch.allclose(packed, ref, atol=1e-5)
    transposed = []
    for img, (h, w) in zip(images, hw.tolist()):
        px = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        z = _official_encode(stub, px)[0]
        transposed += [z[:, y, x] for x in range(w) for y in range(h)]
    must_fail(
        lambda: _check(torch.allclose(packed, torch.stack(transposed), atol=1e-5)),
        "order",
    )
    with pytest.raises(ValueError):
        vae.encode_images(images, torch.tensor([[3, 3], [4, 2]]))


def test_vae_encoder_factory_returns_encode_callable(monkeypatch):
    stub = StubVae()
    monkeypatch.setattr(
        vae_module.AutoencoderKLFlux2,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: stub),
    )
    kwargs = {"path": "unused", "device": "cpu", "dtype": "fp32"}
    encode = build_encoder("kohakumoss.encoders.vae_encoder", kwargs)
    images = [np.zeros((32, 32, 3), dtype=np.uint8)]
    out = encode(images, torch.tensor([[2, 2]]))
    assert out.shape == (4, 128)
    assert not isinstance(encode, nn.Module)

    def module_factory(**kw):
        return FluxVae.from_pretrained(kw["path"], kw["device"], torch.float32)

    wrong = module_factory(**kwargs)
    with pytest.raises(NotImplementedError):
        wrong(images, torch.tensor([[2, 2]]))
    assert vae_encoder("unused", "cpu", torch.float32).__self__.vae is stub
