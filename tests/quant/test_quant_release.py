"""Quantized bases are freed by reference counting alone, model-free (small real
weights on the host): with the cyclic collector off, `del` kills an int8 / fp8 base
bound for its GEMM, its `q` buffer and the MossLinear or QuantLinear over it, and an
fp8-compute base before its first call; the bound GEMM matches the eager quant op. A
base whose `linear` is a bound method of itself stays alive."""

import types
from functools import partial

import pytest
import torch
from torch import nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.compute import ComputeWeight
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.weights import QuantWeight
from kohakumoss_kernels.quant import quant_linear_eager
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu
OUT, IN = 256, 256


def _weight() -> torch.Tensor:
    return torch.randn(OUT, IN, generator=torch.Generator().manual_seed(0)) * 0.1


def _stored(fmt: str, bound: bool = False):
    """`build()` of a MossLinear over a bound `fmt` base (optionally its `linear` a
    bound method of the base)."""

    def build() -> tuple:
        qw = QuantWeight.from_weight(_weight(), fmt)
        qw.bind(torch.float32)
        if bound:
            vars(qw)["linear"] = types.MethodType(QuantWeight.linear, qw)
        lin = MossLinear(IN, OUT, bias=False)
        lin.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        lin.set_base(qw)
        x = torch.randn(3, IN)
        want = quant_linear_eager(x, qw.q, qw.scale, None, fmt)
        assert torch.equal(lin(x), want)
        return lin, qw, qw.q

    return build


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_stored_base_freed_on_del(fmt):
    check_released(_stored(fmt))
    must_fail(lambda: check_released(_stored(fmt, True)), "a self-bound base linear")


def _quant_linear(bound: bool = False):
    def build() -> tuple:
        plain = nn.Linear(IN, OUT)
        ql = QuantLinear(plain, "int8", torch.float32)
        if bound:
            ql._impl = types.MethodType(QuantLinear._with_bias, ql)
        return ql, ql.qweight, ql.bias

    return build


def test_quant_linear_freed_on_del():
    check_released(_quant_linear())
    must_fail(lambda: check_released(_quant_linear(True)), "a bound-method _impl")


def _compute(bound: bool = False):
    def build() -> tuple:
        cw = ComputeWeight.from_weight(_weight(), "row")
        if bound:
            cw._linear = partial(ComputeWeight._unbound.__get__(cw))
        return cw, cw.q

    return build


def test_compute_base_freed_on_del():
    check_released(_compute())
    must_fail(lambda: check_released(_compute(True)), "a partial over self")
