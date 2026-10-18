"""Adapters on the real backbone over a 16-bit, an int8 weight-only and an fp8-compute
base: LoRA and LoKr (full matrix, factor 16) bind as bypass, a forward / backward step
trains only the adapter (finite, non-zero gradients) with the base storage untouched,
one adapted layer (its adapter amplified so that dropping the delta breaks the check)
matches the fp64 formula with error statistics no worse than the eager composition's,
and the LyCORIS export
re-imported for inference (merge on the 16-bit base, bypass on the quantized ones)
reproduces it."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.adapters import AdapterConfig, AdapterStack, apply
from kohakumoss.model import load_backbone
from kohakumoss.quant.convert import quantize_model
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
HWS = ((12, 16), (8, 20))
TEXT_LENS = (21, 33)
LAYER = "layers.3.proj.q"
SPECS = {
    "lora": dict(algo="lora", dim=8, alpha=8.0),
    "lokr": dict(
        algo="lokr",
        dim=10000,
        alpha=1.0,
        algo_kwargs={"factor": 16, "full_matrix": True},
    ),
}


def _fp8_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


BASES = [
    "compute",
    "int8",
    pytest.param(
        "fp8_compute",
        marks=pytest.mark.skipif(not _fp8_ok(), reason="fp8 compute needs sm89+"),
    ),
]


def backbone(weights_root, base: str):
    m, report = load_backbone(
        weights_root / "base", device="cuda", dtype=torch.bfloat16
    )
    assert report.ok
    if base != "compute":
        quantize_model(m, base, compute_dtype=torch.bfloat16)
    return m.requires_grad_(False)


def batch(model):
    cfg = model.cfg
    g = torch.Generator().manual_seed(1)
    img = torch.randn(sum(h * w for h, w in HWS), cfg.latent_channels, generator=g)
    text = torch.randn(sum(TEXT_LENS), cfg.text_dim, generator=g)
    seq = build_seqinfo(torch.tensor(HWS), torch.tensor(TEXT_LENS)).to("cuda")
    dev = torch.device("cuda")
    sigma = torch.tensor([0.3, 0.7], device=dev)
    return img.to(dev, torch.bfloat16), text.to(dev, torch.bfloat16), sigma, seq


def perturb(aset, seed: int = 4) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in aset.named_trainable():
            p.add_(torch.randn(p.shape, generator=g).to(p) * 0.02)


def amplify(lyco, seed: int = 5, std: float = 0.1) -> None:
    """Noise of `std` on one layer's adapter tensors (a delta about the size of the
    base weight)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in lyco.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g).to(p) * std)


def layer_delta(lyco) -> torch.Tensor:
    """dW of the layer's adapter in fp64 from its tensors."""
    if hasattr(lyco, "lora_up"):
        dw = lyco.lora_up.weight.double() @ lyco.lora_down.weight.double()
    else:
        dw = torch.kron(lyco.lokr_w1.double(), lyco.lokr_w2.double())
    return dw * float(lyco.scale)


def layer_refs(lin, lyco, x) -> tuple[torch.Tensor, torch.Tensor]:
    """(PyTorch's bf16 output, the fp64 ruler) of base + adapter at the layer; the base
    is read once, here, as the reference."""
    w = lin.base_weight(torch.float32).double() + layer_delta(lyco)
    b = None if lin.bias is None else lin.bias.double()
    ref = F.linear(x.double(), w, b)
    eager = F.linear(x, w.to(x.dtype), None if b is None else b.to(x.dtype))
    return eager, ref


def base_only(lin, x) -> torch.Tensor:
    bias = None if lin.bias is None else lin.bias.to(x.dtype)
    return F.linear(x, lin.base_weight(x.dtype), bias)


@pytest.mark.parametrize("base", BASES)
def test_bypass_train_export_import(weights_root, base):
    model = backbone(weights_root, base)
    lin = model.get_submodule(LAYER)
    stored = lin.base is not None
    img, text, sigma, seq = batch(model)
    x = torch.randn(64, lin.in_features, device="cuda", dtype=torch.bfloat16)
    for name, fields in SPECS.items():
        config = AdapterConfig(preset="attn-mlp", **fields)
        aset = apply(model, config, device="cuda")
        perturb(aset)
        model.bind_ops()
        model(img, text, sigma, seq).float().square().mean().backward()
        grads = [p.grad for p in aset.trainable_params()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(g.abs().sum() > 0 for g in grads), name
        if stored:
            assert lin.weight.numel() == 0, "a dense base weight was materialized"
        target = next(t for t in aset.targets if t.path == LAYER)
        amplify(target.lyco)
        eager, ref = layer_refs(lin, target.lyco, x)
        with torch.no_grad():
            trained = lin(x)
        assert_no_regression(
            trained, eager, ref, name=f"{base}-{name}", test="adapters"
        )
        must_fail(
            lambda: assert_no_regression(
                base_only(lin, x), eager, ref, test="adapters"
            ),
            "the adapter's delta dropped",
        )
        tensors = aset.tensors("lycoris")
        aset.remove()
        stack = AdapterStack(model, mode="bypass" if stored else "merge")
        stack.add(tensors, 1.0)
        model.bind_ops()
        with torch.no_grad():
            loaded = lin(x)
        assert_no_regression(
            loaded, eager, ref, name=f"reload-{base}-{name}", test="adapters"
        )
        stack.remove()
        model.bind_ops()
