"""apply_fp16_fix keeps one consistent state: idempotent, conflict-checked, clearable."""

import copy

import pytest
import torch

from kohakumoss.model.numerics import apply_fp16_fix, clear_fp16_fix
from tests._util.negative import must_fail
from tests.model.test_fp16fix import _inputs, _model, _rel

pytestmark = pytest.mark.cpu


def _eps(model):
    return {n: m.eps for n, m in model.named_modules() if hasattr(m, "eps")}


def _check(cond):
    assert cond


def test_same_fix_twice_is_a_no_op():
    model = _model()
    x = _inputs(model.cfg, torch.float64)
    ref = model(*x)
    apply_fp16_fix(model, "online", c=2.0**-4)
    eps = _eps(model)
    apply_fp16_fix(model, "online", c=2.0**-4)
    assert _eps(model) == eps
    assert _rel(model(*x), ref) < 1e-12


def test_conflicting_fix_raises_and_clear_restores():
    base = _model()
    x = _inputs(base.cfg, torch.float64)
    ref = base(*x)
    model = copy.deepcopy(base)
    apply_fp16_fix(model, "online", c=2.0**-4)
    with pytest.raises(ValueError):
        apply_fp16_fix(model, "online", c=2.0**-6)
    with pytest.raises(ValueError):
        apply_fp16_fix(model, "weights", c=2.0**-4)
    clear_fp16_fix(model)
    assert _eps(model) == _eps(base) and model.fp16_fix == "none"
    apply_fp16_fix(model, "weights", c=2.0**-4)
    assert _rel(model(*x), ref) < 1e-12
    clear_fp16_fix(model)
    for k, v in base.state_dict().items():
        assert torch.equal(model.state_dict()[k], v), k


def test_none_keeps_an_existing_fix():
    model = _model()
    apply_fp16_fix(model, "weights", c=2.0**-3)
    apply_fp16_fix(model, "none")
    assert model.fp16_fix == "weights" and model.fold_scales[0] == 2.0**-3


def test_unguarded_double_online_compounds_eps():
    model = _model()
    x = _inputs(model.cfg, torch.float64)
    ref = model(*x)
    apply_fp16_fix(model, "online", c=2.0**-12)
    model.fp16_fix = "none"
    apply_fp16_fix(model, "online", c=2.0**-12)
    must_fail(lambda: _check(_rel(model(*x), ref) < 1e-8), "eps compounded")
