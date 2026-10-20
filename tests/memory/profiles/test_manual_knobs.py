"""Manual memory mode and the profile catalog: every top-level memory knob reaches the
plan under MEMORY_PROFILE='manual', a named profile together with a memory knob is an
error, and the catalog holds every combination of its axes."""

from itertools import product

import pytest

from kohakumoss.inference import knobs as inference_knobs
from kohakumoss.memory import knob_map
from kohakumoss.memory.profiles import CATALOG_AXES, catalog, resolve
from kohakumoss.training import knobs as train_knobs
from kohakumoss.training.knobs import memory_overrides, validate
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
SETTINGS = {
    "CHECKPOINT": ("sub-block", {}),
    "STREAM": ("ring", {"STREAM_RESIDENT": 3}),
    "STREAM_RESIDENT": (5, {"STREAM": "ring"}),
    "PINNED": (False, {}),
    "ACT_OFFLOAD": (True, {}),
    "OPTIM_STATE": ("stream", {"FUSED_STEP": True}),
    "GRAD_OFFLOAD": (True, {"FUSED_STEP": True}),
    "FUSED_STEP": (True, {}),
}


def _plan(**knobs):
    k = runnable(MEMORY_PROFILE="manual", **knobs)
    return resolve("manual", memory_overrides(k), "cpu")


def _round_trip() -> None:
    assert set(SETTINGS) == set(knob_map.MEMORY_KNOBS)
    for name, (value, partners) in SETTINGS.items():
        plan = _plan(**{name: value, **partners})
        field = knob_map.MEMORY_KNOBS[name]
        assert getattr(plan, field) == value, (name, getattr(plan, field))


def test_manual_round_trips_every_knob(monkeypatch):
    _round_trip()
    assert validate(runnable(MEMORY_PROFILE="manual", CHECKPOINT="none")) == []
    monkeypatch.setattr(train_knobs, "plan_fields", lambda k, profile: {})
    must_fail(_round_trip, "memory knobs dropped on the way to the plan")


def test_precision_knobs_apply_under_any_profile():
    k = runnable(MEMORY_PROFILE="balanced", BASE_PRECISION="int8")
    plan = resolve("balanced", memory_overrides(k), "cpu")
    assert plan.base_precision == "int8" and validate(k) == []


def _mixing_rejected() -> None:
    problems = validate(runnable(MEMORY_PROFILE="balanced", CHECKPOINT="none"))
    assert any("MEMORY_PROFILE='balanced'" in m and "CHECKPOINT" in m for m in problems)


def test_profile_with_a_memory_knob_is_an_error(monkeypatch):
    _mixing_rejected()
    monkeypatch.setattr(train_knobs, "mixing_problems", lambda *a, **kw: [])
    must_fail(_mixing_rejected, "a profile silently overridden by a memory knob")


def test_inference_manual_and_mixing():
    k = {"MEMORY_PROFILE": "manual", "STREAM": "ring", "STREAM_RESIDENT": 4}
    memory, overrides = inference_knobs.memory_of(k)
    assert memory == "manual" and overrides == {"stream": "ring", "stream_resident": 4}
    mixed = knob_map.mixing_problems(
        {"MEMORY_PROFILE": "balanced", "STREAM": "ring"},
        "balanced",
        "MEMORY_PROFILE",
        inference_knobs.INFERENCE_KNOBS,
    )
    assert mixed and "STREAM" in mixed[0]


def assert_resident_mixing(problems_of) -> None:
    """MEMORY_PROFILE='' (the resident default) with a placement knob is an error, not a
    silently ignored knob; unset knobs pass."""
    knobs = inference_knobs.INFERENCE_KNOBS
    mixed = problems_of(
        {"MEMORY_PROFILE": "", "STREAM": "ring"}, "", "MEMORY_PROFILE", knobs
    )
    assert mixed and "resident" in mixed[0] and "STREAM" in mixed[0]
    assert problems_of({"MEMORY_PROFILE": ""}, "", "MEMORY_PROFILE", knobs) == []


def _resident_exempt(k, profile, profile_knob, knobs):
    if profile == "":
        return []
    return knob_map.mixing_problems(k, profile, profile_knob, knobs)


def test_resident_default_with_a_placement_knob_is_an_error():
    assert_resident_mixing(knob_map.mixing_problems)
    must_fail(lambda: assert_resident_mixing(_resident_exempt), "STREAM ignored")


def _names(parts) -> str:
    return "-".join(part for part in parts if part)


def _full_product() -> set[str]:
    return {_names(combo) for combo in product(*(axis for axis in CATALOG_AXES))}


def test_catalog_is_the_full_product():
    """No combination of the axes is excluded (streaming no longer needs
    checkpointing), so the catalog is the whole product."""

    def check(names: set[str]) -> None:
        assert names == _full_product()

    check(set(catalog()))
    hand_picked = set(catalog()) - {"nockpt-ring2-actoff-optcpu-gradoff"}
    must_fail(lambda: check(hand_picked), "a hand-picked catalog missing one entry")
