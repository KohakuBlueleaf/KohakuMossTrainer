"""Knob validation: the known set equals the train script, every problem in one exit."""

import ast
from pathlib import Path

import pytest

from kohakumoss.training.knobs import KNOWN, check, validate

pytestmark = pytest.mark.cpu
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train" / "moss.py"


def script_defaults() -> dict:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                out[name] = eval(
                    compile(ast.Expression(node.value), str(SCRIPT), "eval"), {}
                )
    return out


def runnable(**overrides) -> dict:
    k = script_defaults()
    k.update(
        WEIGHTS="random",
        VARIANT="tiny",
        DATASETS=[{"source": "synthetic"}],
        PRECISION="fp32",
    )
    k.update(overrides)
    return k


def test_known_matches_script():
    assert set(script_defaults()) == KNOWN


def test_valid_defaults_pass():
    assert validate(runnable()) == []


def test_all_problems_in_one_exit():
    k = runnable(LR=-1.0, ALGO="nope", LOGGER="x", LRR=1.0, GRAD_ACC=0)
    with pytest.raises(SystemExit) as err:
        check(k)
    msg = str(err.value)
    for needle in (
        "LR must be positive",
        "ALGO='nope'",
        "LOGGER='x'",
        "did you mean LR",
        "GRAD_ACC",
    ):
        assert needle in msg, needle


def test_cross_knob_conflicts():
    problems = validate(
        runnable(
            MEMORY_OVERRIDES={"fused_step": True},
            DRIVER="lightning",
            PARALLEL="fsdp",
            DEVICES=(0, 1),
        )
    )
    joined = "\n".join(problems)
    assert "fused_step" in joined and "DRIVER='lightning'" in joined
    profile_fused = validate(runnable(MEMORY_PROFILE="low", OPTIMIZER="muon"))
    assert any("muon" in p for p in profile_fused)
