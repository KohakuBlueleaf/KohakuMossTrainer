"""The fp16 'needs a fix' message offers only the FP16_FIX knob values (none / online /
fp32_residual), never the file-recorded 'weights' mode (model-free)."""

import pytest

from kohakumoss.model.numerics import fix_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
KNOB_VALUES = ("online", "fp32_residual")


def check_message(problems) -> None:
    (line,) = problems("none", {}, True, False, "PRECISION='fp16'", "WEIGHTS")
    for value in KNOB_VALUES:
        assert repr(value) in line, line
    assert "'weights'" not in line, line


def _old(fix, kwargs, fp16, carried, subject, file_knob):
    return [f"{subject} needs FP16_FIX ('online', 'weights' or 'fp32_residual')"]


def test_fix_message_offers_knob_values_only():
    check_message(fix_problems)
    must_fail(lambda: check_message(_old), "'weights' offered as a knob value")
