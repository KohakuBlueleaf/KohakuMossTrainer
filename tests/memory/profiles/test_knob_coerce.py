"""Memory knobs given as command-line strings reach the plan as the field's type, and a
plan field of the wrong type is one clear validation line (model-free)."""

from dataclasses import replace

import pytest

from kohakumoss.memory.knob_map import MEMORY_KNOBS, knob_fields
from kohakumoss.memory.profiles import validate
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_coerced(fields: dict) -> None:
    assert fields["stream_resident"] == 4 and type(fields["stream_resident"]) is int
    assert fields["pinned"] is False and fields["act_offload"] is True
    assert fields["stream"] == "ring" and fields["checkpoint"] == "partial:0.2"


def test_command_line_strings_take_the_field_type():
    given = {
        "STREAM_RESIDENT": "4",
        "PINNED": "false",
        "ACT_OFFLOAD": "True",
        "STREAM": "ring",
        "CHECKPOINT": "partial:0.2",
    }
    check_coerced(knob_fields(given, MEMORY_KNOBS))
    raw = {MEMORY_KNOBS[k]: v for k, v in given.items()}
    must_fail(lambda: check_coerced(raw), "strings passed through")


def check_reported(plan: MemoryPlan) -> None:
    problems = validate(plan)
    assert problems == ["stream_resident='four' is not int"], problems


def test_wrong_type_is_one_line():
    check_reported(replace(MemoryPlan(), stream="ring", stream_resident="four"))
    must_fail(lambda: check_reported(MemoryPlan()), "a valid plan")
