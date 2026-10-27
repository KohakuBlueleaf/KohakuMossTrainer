"""Every example run config holds only knobs the trainer knows (its layers resolved)."""

from pathlib import Path

import pytest

from kohakumoss.cli.config.validate import load_knobs
from kohakumoss.training.knobs import KNOWN
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

EXAMPLES = Path(__file__).parents[2] / "examples"
RUNS = sorted(EXAMPLES.glob("*/train_*.py"))


def check(knobs: dict) -> None:
    assert not set(knobs) - KNOWN, sorted(set(knobs) - KNOWN)


@pytest.mark.parametrize("path", RUNS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_example_runs_hold_known_knobs(path):
    knobs = load_knobs(path)
    check(knobs)
    must_fail(lambda: check({**knobs, "PRE": ["artist"]}), "an unknown UPPER_CASE name")


def test_examples_are_found():
    assert RUNS
