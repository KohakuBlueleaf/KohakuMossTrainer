"""Quantized base storage: layout witnessed against an independent formula."""

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.comfy import attach_quantized, split_quantized
from kohakumoss.quant.convert import QuantLinear, quantize_model, quantize_torch_linears
from kohakumoss.quant.weights import QuantWeight, quantize
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _w(out=48, inp=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out, inp, generator=g) * torch.linspace(0.1, 3, out)[:, None]


def _int8_reference(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amax = np.maximum(np.abs(w).max(axis=1), 1e-12).astype(np.float32)
    scale = (amax / np.float32(127.0)).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def test_int8_bytes_match_independent_formula():
    w = _w()
    q, scale = quantize(w, "int8")
    rq, rs = _int8_reference(w.numpy())
    assert (
        hashlib.sha1(q.numpy().tobytes()).hexdigest()
        == hashlib.sha1(rq.tobytes()).hexdigest()
    )
    assert np.allclose(scale.numpy(), rs)

    def check_wrong_axis():
        q2, _ = quantize(w.t().contiguous(), "int8")
        assert np.array_equal(q2.t().numpy(), rq)

    must_fail(check_wrong_axis, "per-column scales are a different layout")


def test_fp8_error_bounded_and_dequant_close():
    w = _w()
    qw = QuantWeight.from_weight(w, "fp8")
    assert qw.q.dtype == torch.float8_e4m3fn
    rel = (qw.materialize() - w).abs() / w.abs().amax(1, keepdim=True)
    assert rel.max().item() <= 2**-4 + 1e-6


def test_quantize_model_replaces_and_forward_close():
    lin = MossLinear(512, 384, bias=True)
    lin.init_weights(0.05)
    x = torch.randn(7, 512)
    ref = lin(x)
    holder = nn.ModuleDict({"lin": lin})
    report = quantize_model(holder, "int8")
    assert report["layers"] == 1 and report["quant_bytes"] < report["dense_bytes"] / 3
    assert isinstance(lin.base, QuantWeight) and lin.weight.numel() == 0
    err = (lin(x) - ref).norm() / ref.norm()
    assert err < 1e-2


def test_comfy_quantized_state_loads():
    w = _w(512, 384)
    q, scale = quantize(w, "fp8")
    state = {
        "blk.lin.weight": q,
        "blk.lin.scale_weight": scale.max().reshape(()),
        "x.bias": torch.ones(3),
    }
    quant, dense = split_quantized(state)
    assert set(dense) == {"x.bias"} and "blk.lin" in quant
    assert quant["blk.lin"].scale.shape == (512,)
    model = nn.Module()
    model.blk = nn.Module()
    model.blk.lin = MossLinear(384, 512, bias=False)
    assert attach_quantized(model, quant, torch.float32) == 1
    assert model.blk.lin.base is quant["blk.lin"]


def test_quantize_torch_linears_on_plain_modules():
    m = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 300))
    x = torch.randn(3, 256)
    ref = m(x)
    assert quantize_torch_linears(m, "int8", torch.float32) == 2
    assert isinstance(m[0], QuantLinear)
    assert ((m(x) - ref).norm() / ref.norm()) < 2e-2
