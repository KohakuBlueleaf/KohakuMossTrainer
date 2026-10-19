"""Quantized bases are freed by reference counting alone, model-free (small real
weights on the host): with the cyclic collector off, `del` kills an int8 / fp8 base
bound for its GEMM, its `q` buffer and the MossLinear or QuantLinear over it, and an
fp8-compute base before its first call, and the ComfyUI kinds (a ConvRot int8 base, a
W4A8 base, an int8 embedding) after a call; the bound GEMM matches the eager quant op. A
base holding a bound method of itself stays alive."""

import types
from functools import partial

import pytest
import torch
from torch import nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.compute import ComputeWeight
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.w4a8 import W4A8Weight
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


def _comfy(kind: str, bound: bool = False):
    """`build()` of a ConvRot int8 base, a W4A8 base (both bound for their GEMM) or an
    int8 embedding (bound for its rotation), optionally holding a bound method of
    itself."""

    def build() -> tuple:
        g = torch.Generator().manual_seed(1)
        q = torch.randint(-127, 128, (OUT, IN), generator=g, dtype=torch.int8)
        scale = torch.rand(OUT, generator=g) * 1e-2
        match kind:
            case "convrot":
                base = QuantWeight(q, scale, "int8", 256)
            case "w4a8":
                packed = torch.randint(-128, 128, (OUT, IN // 2), generator=g)
                s_rel = (torch.rand(OUT, IN // 16, generator=g) + 1).to(
                    torch.float8_e4m3fn
                )
                base = W4A8Weight(
                    packed.to(torch.int8), s_rel, scale, torch.zeros(16), 16, 256
                )
            case _:
                base = QuantEmbedding(q, scale, 256, torch.float32)
        base.bind(torch.float32)
        if bound:
            vars(base)["held"] = types.MethodType(type(base).forward, base)
        if kind == "embedding":
            base(torch.tensor([0, 3]))
        else:
            base.linear(torch.randn(2, IN), None)
        return base, base.q

    return build


@pytest.mark.parametrize("kind", ["convrot", "w4a8", "embedding"])
def test_comfy_bases_freed_on_del(kind):
    check_released(_comfy(kind))
    must_fail(lambda: check_released(_comfy(kind, True)), "a self-bound method")
