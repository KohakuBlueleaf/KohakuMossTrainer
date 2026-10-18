"""An invalid value of a knob with a long choice list (MEMORY_PROFILE, ~240 catalog
names) gets the 3 closest choices and where the full list is, not the whole list; a
short list is printed in full (model-free)."""

import pytest

from kohakumoss.inference import knobs as sample_knobs
from kohakumoss.training import knobs as train_knobs
from kohakumoss_common.knobs import choice_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
POINTER = "kohakumoss-config list --all"


def check_message(problems) -> None:
    for choices in (train_knobs.CHOICES, sample_knobs.CHOICES):
        (line,) = problems({"MEMORY_PROFILE": "balansed"}, choices)
        assert POINTER in line, line
        named = line.split("did you mean ")[1].split("?)")[0].split(", ")
        assert 1 <= len(named) <= 3, (len(named), line)
        assert "balanced" in named, line
    (short,) = problems({"DRIVER": "nativ"}, train_knobs.CHOICES)
    assert "('native', 'lightning')" in short, short


def _full_dump(k: dict, choices: dict) -> list[str]:
    return [
        f"{n}={k[n]!r} not in {opts}"
        for n, opts in choices.items()
        if n in k and k[n] not in opts
    ]


def test_long_choice_lists_name_three_and_point_to_the_list():
    check_message(choice_problems)
    must_fail(lambda: check_message(_full_dump), "the whole catalog in one line")
