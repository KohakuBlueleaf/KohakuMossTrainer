"""The fp16 'no fix' warning offers only the FP16_FIX knob values (none / online /
fp32_residual), never the file-recorded 'weights' mode, and fires only for fp16 compute
with FP16_FIX='none' on weights without a fold (model-free)."""

import pytest

from kohakumoss.model.numerics import fix_problems, fix_warnings
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
KNOB_VALUES = ("online", "fp32_residual")


def check_message(warnings) -> None:
    (line,) = warnings("none", True, False, "PRECISION='fp16'", "WEIGHTS")
    for value in KNOB_VALUES:
        assert repr(value) in line, line
    assert "'weights'" not in line, line


def _old(fix, fp16, carried, subject, file_knob):
    return [f"{subject} needs FP16_FIX ('online', 'weights' or 'fp32_residual')"]


def test_fix_message_offers_knob_values_only():
    check_message(fix_warnings)
    must_fail(lambda: check_message(_old), "'weights' offered as a knob value")


def test_warning_fires_only_on_unfixed_fp16():
    assert fix_warnings("none", True, False, "s", "WEIGHTS")
    assert not fix_warnings("none", False, False, "s", "WEIGHTS")
    assert not fix_warnings("none", True, True, "s", "WEIGHTS")
    for fix in KNOB_VALUES:
        assert not fix_warnings(fix, True, False, "s", "WEIGHTS")


def test_unfixed_fp16_is_not_a_problem():
    assert fix_problems("none", {}) == []
    assert fix_problems("online", {})
    assert fix_problems("online", {"c": 1 / 16}) == []
