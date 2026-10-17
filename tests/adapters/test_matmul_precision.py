"""Adapter matmuls KohakuMoss runs itself take 16-bit operands: on a conditioning
linear (MossLinear with an input dtype, fp32 weights and inputs) the dense dW and the
merged-weight matmuls go through the layer's 16-bit-input, fp32-accumulating GEMM."""

import pytest
import torch
import torch.nn.functional as F
from _adcases import build, make_spec
from _adtoy import perturb, toy

from kohakumoss.adapters import apply, load_adapters
from kohakumoss.model.components.wide import wide_linear
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
LAYER = "shared_block_modulation"
HALF = torch.bfloat16


def _wide_toy():
    model = toy()
    model.get_submodule(LAYER).set_input_dtype(HALF)
    model.eval()
    return model


def _target(aset):
    return next(t for t in aset.targets if t.path == LAYER)


def _x(t):
    return torch.randn(5, t.lyco.shape[1], generator=torch.Generator().manual_seed(3))


def _adapted(name, **kw):
    model = _wide_toy()
    torch.manual_seed(0)
    aset = apply(model, make_spec(name, preset="attn+modulation", **kw))
    perturb(aset)
    return aset


def _dense_bypass_ok(aset, dense):
    t = _target(aset)
    x = _x(t)
    with torch.no_grad():
        dw = t.delta(0.5)[0]
        want = t.linear._impl(x) + dense(x, dw)
        torch.testing.assert_close(t.linear(x), want, rtol=0, atol=0)


def test_dense_bypass_uses_the_layer_gemm():
    aset = _adapted("oft")
    aset.set_multiplier(0.5)
    _dense_bypass_ok(aset, lambda x, w: wide_linear(x, w, HALF))

    def fp32():
        _dense_bypass_ok(aset, lambda x, w: F.linear(x, w))

    must_fail(fp32, "an fp32 x @ dW^T on a 16-bit-input layer")


def _merged_ok(aset, lin):
    t = _target(aset)
    x = _x(t)
    with torch.no_grad():
        w = t.base_weight(torch.float32) + t.delta(1.0)[0]
        want = lin(x, w) + t.linear.bias.float()
        torch.testing.assert_close(t.linear(x), want, rtol=0, atol=0)


def test_merge_mode_uses_the_layer_gemm():
    aset = _adapted("lora", mode="merge", backend="torch")
    _merged_ok(aset, lambda x, w: wide_linear(x, w, HALF))
    must_fail(
        lambda: _merged_ok(aset, lambda x, w: F.linear(x, w)),
        "an fp32 merged-weight matmul on a 16-bit-input layer",
    )


def test_loaded_dense_delta_uses_the_layer_gemm(tmp_path):
    _, source = build("dora-out", preset="attn+modulation")
    path = tmp_path / "d.safetensors"
    source.export(str(path), "comfy", dtype=None)
    model = _wide_toy()
    layer = model.get_submodule(LAYER)
    x = torch.randn(5, layer.in_features)
    with torch.no_grad():
        base = layer(x)
    load_adapters(model, [(str(path), 1.0)], "bypass")
    fn = vars(layer)["forward"]
    dw = fn.args[2]
    assert dw.dtype == HALF
    with torch.no_grad():
        torch.testing.assert_close(
            layer(x), base + wide_linear(x, dw, HALF), rtol=0, atol=0
        )
