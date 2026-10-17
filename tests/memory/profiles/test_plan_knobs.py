"""MemoryPlan problems are knob problems: resolved at validation (one exit, before any
weight load), with the PARALLEL / OPTIMIZER / COMPILE combinations that cannot run."""

import pytest

from kohakumoss.training import knobs as knobs_mod
from kohakumoss.training.entry import run
from kohakumoss.training.knobs import plan_problems, validate
from kohakumoss.training.optim import factory as optim_factory
from tests.training.test_entry_smoke import smoke_knobs
from tests.training.test_knobs import runnable

pytestmark = pytest.mark.cpu
MULTI = dict(DEVICES=(0, 1))


def _has(problems: list[str], *needles: str) -> bool:
    return all(any(n in p for p in problems) for n in needles)


CASES = {
    "plan_resolved_at_validation": (
        dict(
            MEMORY_OVERRIDES={
                "stream": "ring",
                "checkpoint": "none",
                "optim_state": "x",
            }
        ),
        ("every block checkpointed", "optim_state 'x'"),
    ),
    "base_precision_16bit_removed": (
        dict(MEMORY_OVERRIDES={"base_precision": "fp16"}),
        ("base_precision 'fp16'",),
    ),
    "compile_is_a_knob": (
        dict(MEMORY_OVERRIDES={"compile": "block"}),
        ("set the COMPILE knob",),
    ),
    "optim_state_needs_adamw": (
        dict(MEMORY_OVERRIDES={"optim_state": "stream"}, OPTIMIZER="adamw16"),
        ("optim_state='stream' places AdamW states",),
    ),
    "swap_fused_contradiction": (
        dict(
            ALGO="full",
            MEMORY_OVERRIDES={
                "stream": "swap",
                "stream_resident": 2,
                "fused_step": False,
            },
        ),
        ("stream='swap' steps blocks inside backward",),
    ),
    "ddp_swap": (
        dict(
            ALGO="full",
            PARALLEL="ddp",
            MEMORY_OVERRIDES={"stream": "swap", "stream_resident": 2},
            **MULTI,
        ),
        ("does not reduce streamed",),
    ),
    "fsdp_stream_states": (
        dict(
            PARALLEL="fsdp",
            DIST_BACKEND="nccl",
            MEMORY_OVERRIDES={"optim_state": "stream"},
            **MULTI,
        ),
        ("PARALLEL='fsdp' needs optim_state='gpu'",),
    ),
    "fsdp_act_offload": (
        dict(
            PARALLEL="fsdp",
            DIST_BACKEND="nccl",
            MEMORY_OVERRIDES={"act_offload": True},
            **MULTI,
        ),
        ("needs act_offload=False",),
    ),
    "fsdp_gloo": (
        dict(PARALLEL="fsdp", DIST_BACKEND="gloo", **MULTI),
        ("PREMUL_SUM",),
    ),
    "fsdp_sampling": (
        dict(
            PARALLEL="fsdp",
            DIST_BACKEND="nccl",
            SAMPLE_EVERY=10,
            SAMPLE_PROMPTS=["x"],
            **MULTI,
        ),
        ("set SAMPLE_EVERY=0",),
    ),
    "sub_block_with_compile": (
        dict(COMPILE="block", MEMORY_OVERRIDES={"checkpoint": "sub-block"}),
        ("checkpoint='sub-block' calls block parts directly",),
    ),
    "pipeline_stream": (
        dict(
            PARALLEL="threaded",
            MEMORY_OVERRIDES={"stream": "ring", "stream_resident": 2},
            **MULTI,
        ),
        ("PARALLEL='threaded' needs stream='off'",),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_plan_problem_reported(name):
    knobs, needles = CASES[name]
    problems = validate(runnable(**knobs))
    assert _has(problems, *needles), problems


@pytest.mark.parametrize("name", list(CASES))
def test_plan_problem_absent_without_the_knob(name):
    """The message names the combination: the default knobs do not produce it."""
    _, needles = CASES[name]
    assert not _has(validate(runnable()), *needles)


def test_8bit_states_need_bitsandbytes(monkeypatch):
    """optim_state='8bit' without the package is a knob problem, and `auto` skips the
    profiles that need it."""
    monkeypatch.setattr(optim_factory, "bnb", None)
    needle = "optim_state='8bit' needs the bitsandbytes package"
    assert _has(validate(runnable(MEMORY_PROFILE="low")), needle)
    assert not _has(validate(runnable(MEMORY_PROFILE="balanced")), needle)
    plans, _ = knobs_mod._plans(runnable(MEMORY_PROFILE="auto"))
    runnable_low = [
        q for q in plans if q.profile == "low" and not plan_problems(runnable(), q)
    ]
    assert not runnable_low


@pytest.mark.slow
def test_auto_profile_builds_and_trains(tmp_path):
    knobs = smoke_knobs(tmp_path, MEMORY_PROFILE="auto", STEPS=2)
    assert validate(knobs) == []
    assert run(knobs).step == 2
