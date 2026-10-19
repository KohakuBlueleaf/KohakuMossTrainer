"""A backbone and its linears are freed by reference counting alone, model-free (the
MOSS-Image base architecture on the meta device, ops bound for the host): with the
cyclic collector off, `del` kills the backbone, a linear and its weight; a MossLinear
selects a class function and forwards as F.linear; the text encoder's run slot and its
one-shot first pass hold no reference to the encoder. A bound-method forward slot on
one linear, or a plan bound to the backbone, keeps them alive."""

import types

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.encoders.text.encoder import QwenTextEncoder
from kohakumoss.model import build_backbone, resolve_config
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.opset import OpSet
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")


def _backbone(broken=None):
    """`build()` of the real architecture with host ops; `broken(model)` puts a cycle
    back before the references are dropped."""

    def build() -> tuple:
        model = build_backbone(resolve_config("base"), "meta", torch.bfloat16)
        model.bind_ops(OpSet.resolve(CPU, torch.bfloat16, {}))
        q = model.layers[0].proj["q"]
        if broken is not None:
            broken(model)
        return model, q, q.weight, model.layers[-1]

    return build


def _bound_impl(model) -> None:
    q = model.layers[0].proj["q"]
    q._impl = types.MethodType(MossLinear._dense_nobias, q)


def _bound_plan(model) -> None:
    model._default = model.bind_forward()


def test_backbone_freed_on_del():
    check_released(_backbone())
    must_fail(lambda: check_released(_backbone(_bound_impl)), "a bound-method _impl")
    must_fail(lambda: check_released(_backbone(_bound_plan)), "a plan bound to self")


def check_linear(lin: MossLinear, bias: bool) -> None:
    """The selected slot is a class function and the forward is F.linear."""
    assert isinstance(lin._impl, types.FunctionType), lin._impl
    x = torch.randn(5, lin.in_features)
    want = F.linear(x, lin.weight, lin.bias if bias else None)
    assert torch.equal(lin(x), want)


def test_linear_selects_a_class_function():
    for bias in (True, False):
        lin = MossLinear(8, 4, bias=bias)
        lin.init_weights()
        lin.rebind()
        check_linear(lin, bias)
    lin = MossLinear(8, 4, bias=False)
    lin.init_weights()
    lin._impl = types.MethodType(MossLinear._dense_nobias, lin)
    must_fail(lambda: check_linear(lin, False), "a bound method in the slot")


def _encoder(log: list, closure: bool = False):
    """`build()` of an encoder shell (no model) armed with a one-shot first pass."""

    def build() -> tuple:
        te = object.__new__(QwenTextEncoder)
        te._run = QwenTextEncoder._encode
        warm = torch.zeros(4)

        def fn() -> None:
            log.append(warm.sum())

        if closure:
            run = te._run

            def first(prompts):
                te._run = run
                fn()
                return run(te, prompts)

            te._run = first
        else:
            te.before_first(fn)
        return te, warm

    return build


def test_text_encoder_first_pass_holds_no_encoder():
    check_released(_encoder([]))
    must_fail(lambda: check_released(_encoder([], True)), "a closure over the encoder")
