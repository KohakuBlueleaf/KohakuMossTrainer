"""A `--set` override of a list-of-prompts knob never splits a prompt at its commas: a
bare string is one prompt, list syntax is the list; the sample and grid scripts parse
PROMPTS through prompt_list, through kohakuengine's own coercion."""

import importlib.util
from pathlib import Path

import pytest
from kohakuengine import RawArg
from kohakuengine.engine.coerce.globals import coerce_globals

from kohakumoss_common.knobs import prompt_list
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ("scripts/sample/sample.py", "scripts/sample/compare_grid.py")
SCRIPTS += ("scripts/examples/grid_prompts.py",)
PROMPT = "a lighthouse, rocky coast, sunset"
TRAIN = "scripts/train/moss.py"


def annotations(rel: str) -> dict:
    spec = importlib.util.spec_from_file_location(
        "_prompt_" + Path(rel).stem, ROOT / rel
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__annotations__


def _keeps_commas(parse) -> None:
    assert parse(PROMPT) == [PROMPT]
    assert parse('["a, b", "c"]') == ["a, b", "c"]
    assert parse("[a, b]") == ["a", "b"]
    assert parse(["x, y"]) == ["x, y"]


def test_a_bare_string_is_one_prompt():
    _keeps_commas(prompt_list)
    must_fail(lambda: _keeps_commas(lambda v: v.split(",")), "a comma split")


@pytest.mark.parametrize("rel", SCRIPTS)
def test_the_scripts_parse_prompts_through_it(rel):
    notes = annotations(rel)
    raw, defaults = {"PROMPTS": RawArg(PROMPT)}, {"PROMPTS": []}
    got = coerce_globals(raw, defaults, annotations=notes)
    assert got["PROMPTS"] == [PROMPT], got
    plain = coerce_globals(raw, defaults, annotations={})
    must_fail(lambda: _one(plain["PROMPTS"]), "kohakuengine's default list split")


def _one(prompts: list) -> None:
    assert prompts == [PROMPT], prompts


def test_the_train_script_parses_sample_prompts_through_it():
    notes = annotations(TRAIN)
    raw, defaults = {"SAMPLE_PROMPTS": RawArg(PROMPT)}, {"SAMPLE_PROMPTS": []}
    got = coerce_globals(raw, defaults, annotations=notes)
    assert got["SAMPLE_PROMPTS"] == [PROMPT], got
    plain = coerce_globals(raw, defaults, annotations={})
    must_fail(lambda: _one(plain["SAMPLE_PROMPTS"]), "kohakuengine's list split")
