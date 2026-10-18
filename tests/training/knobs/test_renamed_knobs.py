"""A removed or renamed knob is reported with what replaced it (a rename table and the
encoder dict knobs' fields), not with a fuzzy did-you-mean (model-free)."""

import pytest

from kohakumoss.encoders.knob_names import ENCODER_FIELDS
from kohakumoss.inference import knobs as sample_knobs
from kohakumoss.training import knobs as train_knobs
from kohakumoss_common.knobs import unknown_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OLD = {
    "DTYPE": "renamed to PRECISION",
    "MEMORY": "renamed to MEMORY_PROFILE",
    "RESIDENCY": "TEXT_ENCODE['free_after_use']",
    "TEXT_MEMORY": "now TEXT_ENCODE['memory']",
    "TEXT_PRECISION": "now TEXT_ENCODE['precision']",
    "VAE_TILE": "now VAE_ENCODE['tile']",
}


def check_hints(problems_of) -> None:
    for module in (train_knobs, sample_knobs):
        lines = problems_of(module, dict.fromkeys(OLD, 1))
        for name, text in OLD.items():
            line = next(p for p in lines if p.startswith(f"unknown knob {name}"))
            assert text in line and "did you mean" not in line, line


def _with_tables(module, k: dict) -> list[str]:
    return unknown_problems(k, module.KNOWN, module.RENAMED, ENCODER_FIELDS)


def _fuzzy_only(module, k: dict) -> list[str]:
    return unknown_problems(k, module.KNOWN)


def test_renamed_knobs_name_their_replacement():
    check_hints(_with_tables)
    must_fail(lambda: check_hints(_fuzzy_only), "difflib's guess for a renamed knob")


def test_the_scripts_validate_with_the_tables():
    train = train_knobs._unknown({"DTYPE": "bf16", "SIZES": [(1024, 1024)]})
    assert "unknown knob DTYPE: renamed to PRECISION (the compute dtype)" in train
    assert "unknown knob SIZES: the train script's name is SAMPLE_SIZE" in train
    sample = unknown_problems(
        {"SAMPLE_SEEDS": [0]}, sample_knobs.KNOWN, sample_knobs.RENAMED, ENCODER_FIELDS
    )
    assert sample == ["unknown knob SAMPLE_SEEDS: the sample script's name is SEEDS"]
    assert "did you mean" in unknown_problems({"PRECISON": 1}, train_knobs.KNOWN)[0]
