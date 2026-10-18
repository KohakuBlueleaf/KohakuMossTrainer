"""OPTIMIZER_KWARGS is checked against the chosen builder's own arguments: a name the
builder does not take (a typo, `rounding` on AdamW, LR given twice) is a knob problem;
a dotted-path optimizer is not checked."""

import pytest

from kohakumoss.training.optim.optimizers import kwargs as optim_kwargs
from kohakumoss.training.optim.optimizers.kwargs import builder_kwargs, kwargs_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _k(optimizer, **given) -> dict:
    return {"OPTIMIZER": optimizer, "OPTIMIZER_KWARGS": given}


def check_rejects() -> None:
    assert kwargs_problems(_k("adamw", rounding="kahan"))
    assert kwargs_problems(_k("muon", momemtum=0.9))
    assert kwargs_problems(_k("adamw16", lr=1.0))


def test_builder_arguments():
    assert builder_kwargs("adamw16") == ("rounding",)
    assert set(builder_kwargs("muon")) == {"momentum", "ns_steps"}
    assert builder_kwargs("adamw") == ()
    assert builder_kwargs("my.module.Optimizer") is None
    assert kwargs_problems(_k("adamw16", rounding="kahan")) == []
    assert kwargs_problems(_k("muon", momentum=0.9, ns_steps=3)) == []
    assert kwargs_problems(_k("my.module.Optimizer", anything=1)) == []
    check_rejects()


def test_unchecked_variant_fails(monkeypatch):
    monkeypatch.setattr(optim_kwargs, "builder_kwargs", lambda name: None)
    must_fail(check_rejects, "OPTIMIZER_KWARGS not checked")
