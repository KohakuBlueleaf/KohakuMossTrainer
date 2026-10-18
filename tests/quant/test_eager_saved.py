"""A frozen quantized base on the eager `quant_linear` keeps only `q` and `scale` for
backward (no 16-bit dequantized weight), with the reference forward and input
gradient (model-free: one explicit weight on the host)."""

import math

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.quant.weights import QuantWeight, dequantize
from kohakumoss_kernels.quant import quant_linear_reference
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
N, K = 48, 64


def _saved(linear, x: torch.Tensor) -> list[tuple]:
    seen = []

    def pack(t):
        seen.append((t.dtype, tuple(t.shape)))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        linear(x).float().square().sum().backward()
    return seen


def check_no_weight_copy(linear, x) -> None:
    saved = _saved(linear, x)
    weights = [s for d, s in saved if d == x.dtype and math.prod(s) == N * K]
    assert not weights, saved


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_eager_saves_q_and_scale(fmt):
    torch.manual_seed(0)
    qw = QuantWeight.from_weight(torch.randn(N, K), fmt)
    qw.bind(torch.bfloat16, "cpu")
    x = torch.randn(5, K, dtype=torch.bfloat16, requires_grad=True)
    check_no_weight_copy(lambda v: qw.linear(v, None), x)
    got, x.grad = x.grad.clone(), None
    ref = quant_linear_reference(x, qw.q, qw.scale, None, fmt)
    ref.float().square().sum().backward()
    torch.testing.assert_close(got, x.grad)
    torch.testing.assert_close(qw.linear(x, None), ref)

    def dense(v):
        return F.linear(v, dequantize(qw.q, qw.scale, v.dtype))

    must_fail(lambda: check_no_weight_copy(dense, x), "dequantize + F.linear")
