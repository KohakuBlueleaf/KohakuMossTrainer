"""Adding a bias to a bias-less MossLinear re-selects its bias forward (and removal undoes it)."""

import pytest
import torch
from _adtoy import toy

from kohakumoss.adapters import load_adapters
from kohakumoss.adapters.adapter_set import add_bias
from kohakumoss.model.linear import MossLinear
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
LAYER = "layers.0.proj.q"


def test_add_bias_rebinds_the_forward():
    lin = MossLinear(8, 4, bias=False)
    lin.init_weights(0.2)
    b = torch.randn(4)
    add_bias(lin, b)
    x = torch.randn(3, 8)
    torch.testing.assert_close(lin(x), x @ lin.weight.T + b)


def test_bias_without_rebind_is_caught():
    lin = MossLinear(8, 4, bias=False)
    lin.init_weights(0.2)
    b = torch.randn(4)
    x = torch.randn(3, 8)

    def no_rebind():
        lin.bias = torch.nn.Parameter(b.clone())
        torch.testing.assert_close(lin(x), x @ lin.weight.T + b)

    must_fail(no_rebind, "a bias added without rebind is ignored by the bound forward")


def test_merged_diff_bias_on_a_bias_less_layer_and_exact_removal():
    model = toy()
    layer = model.get_submodule(LAYER)
    assert layer.bias is None
    w0 = layer.weight.detach().clone()
    dw, db = torch.randn_like(w0) * 0.1, torch.randn(w0.shape[0])
    stem = f"diffusion_model.{LAYER}"
    x = torch.randn(5, w0.shape[1])
    with torch.no_grad():
        before = layer(x)
    stack = load_adapters(
        model, [({f"{stem}.diff": dw, f"{stem}.diff_b": db}, 0.5)], mode="merge"
    )
    with torch.no_grad():
        torch.testing.assert_close(layer(x), x @ (w0 + 0.5 * dw).T + 0.5 * db)
    stack.remove()
    assert layer.bias is None
    with torch.no_grad():
        assert torch.equal(layer(x), before)
