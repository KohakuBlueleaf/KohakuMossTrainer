"""Knob validation: the known set equals the train script, every problem in one exit."""

import ast
from pathlib import Path

import pytest

from kohakumoss.training.knobs import KNOWN, check, validate

pytestmark = pytest.mark.cpu
SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "moss.py"
WEIGHTS = str(SCRIPT.parent)  # an existing directory; validation reads no weights


def script_defaults() -> dict:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        match node:
            case ast.Assign(targets=[ast.Name(id=name), *_], value=value):
                pass
            case ast.AnnAssign(target=ast.Name(id=name), value=value) if value:
                pass
            case _:
                continue
        if name.isupper():
            out[name] = eval(compile(ast.Expression(value), str(SCRIPT), "eval"), {})
    return out


def runnable(**overrides) -> dict:
    k = script_defaults()
    k.update(
        WEIGHTS=WEIGHTS,
        VARIANT="base",
        DATASETS=[{"source": "synthetic"}],
        LATENT_CACHE="off",
        TEXT_CACHE="off",
        PRECISION="auto",
        FP16_FIX="online",
        FP16_FIX_KWARGS={"c": 1 / 16},
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
