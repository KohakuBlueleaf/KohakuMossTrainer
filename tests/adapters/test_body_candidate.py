"""The body candidate an autotuned adapter op binds against its fused kernel: the
compiled body, except for bodies through a custom autograd.Function (LoHa's
HadaWeight), whose compiled backward falls below eager's precision: those bind the
eager body and offer no compile candidate."""

import pytest
from lycoris.modules import LoConModule, LohaModule
from torch import nn

from kohakumoss.adapters.bind import bodies, ops, weight_ops
from kohakumoss.adapters.bind.autotuned import AutotunedOp
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
LOHA = (
    weight_ops.loha_weight,
    weight_ops.loha_merged_weight,
    lambda lyco: ops.loha_bypass(lyco, 0.5),
    lambda lyco: ops.loha_bypass_with_base(lyco, 0.5),
)


def _loha():
    return LohaModule("l", nn.Linear(32, 16), 1.0, 4, 4.0)


def _impls(op) -> set[str]:
    return set(AutotunedOp(op, "triton", "tokens").candidates)


def _no_compile(op) -> None:
    assert _impls(op) == {"kernel", "eager"}, _impls(op)


def test_loha_ops_bind_kernel_against_eager():
    for build in LOHA:
        _no_compile(build(_loha()))
    lora = weight_ops.lora_weight(LoConModule("l", nn.Linear(32, 16), 1.0, 4, 4.0))
    assert _impls(lora) == {"kernel", "compile"}


def test_check_catches_a_compiled_loha_candidate(monkeypatch):
    monkeypatch.setattr(bodies, "AUTOGRAD_FUNCTION_BODIES", frozenset())
    must_fail(
        lambda: _no_compile(weight_ops.loha_weight(_loha())), "compiled LoHa body"
    )
