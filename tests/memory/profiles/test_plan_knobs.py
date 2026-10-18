"""MemoryPlan problems are knob problems: resolved at validation (one exit, before any
weight load), with the PARALLEL / OPTIMIZER / COMPILE combinations that cannot run."""

import sys

import pytest

from kohakumoss.memory.profiles import PROFILES
from kohakumoss.training import knobs as knobs_mod
from kohakumoss.training.knobs import plan_problems, validate
from kohakumoss.training.optim.optimizers import factory as optim_factory
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
MULTI = dict(DEVICES=(0, 1))


def _has(problems: list[str], *needles: str) -> bool:
    return all(any(n in p for p in problems) for n in needles)


CASES = {
    "plan_resolved_at_validation": (
        dict(
            MEMORY_OVERRIDES={
                "stream": "ring",
                "stream_resident": 1,
                "optim_state": "x",
            }
        ),
        ("stream_resident >= 2", "optim_state 'x'"),
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
    "fsdp_fused_step": (
        dict(
            PARALLEL="fsdp",
            DIST_BACKEND="nccl",
            MEMORY_OVERRIDES={"fused_step": True},
            **MULTI,
        ),
        ("PARALLEL='fsdp' needs fused_step=False",),
    ),
    "pipeline_grad_offload": (
        dict(
            PARALLEL="threaded",
            MEMORY_OVERRIDES={"grad_offload": True},
            **MULTI,
        ),
        ("PARALLEL='threaded' needs grad_offload=False",),
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


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")


def _added(kind: str, profile: str, **kw) -> list[str]:
    """Plan problems PARALLEL=`kind` over two devices adds to the single-device run."""
    single = runnable(MEMORY_PROFILE=profile, **kw)
    multi = runnable(MEMORY_PROFILE=profile, PARALLEL=kind, **MULTI, **kw)
    plans, _ = knobs_mod._plans(single)
    return [
        m
        for q in plans
        for m in plan_problems(multi, q)
        if m not in plan_problems(single, q)
    ]


def check_every_profile(kind: str, **kw) -> None:
    for name in PROFILES:
        if name != "manual":
            assert _added(kind, name, **kw) == [], name


@pytest.mark.parametrize("algo", ["lora", "full"])
def test_ddp_runs_every_memory_profile(linux, algo):
    """Every memory plan that runs on one device runs under DDP (streaming, offload
    and in-backward steps included); FSDP2 keeps its restrictions. The 8 GB / 6 GB
    and full-finetune streaming profiles validate whole under DDP."""
    kw = {"ALGO": algo}
    if algo == "full":
        kw["EXPORT_FORMATS"] = ("merged",)
    check_every_profile("ddp", **kw)
    must_fail(lambda: check_every_profile("fsdp", **kw), "FSDP2 with streaming")
    profiles = ("minimal", "ckpt-stream2-actoff-optcpu-gradoff")
    if algo == "full":
        profiles = ("ckpt-stream4-actoff-optstream-gradoff", "ckpt-stream4-optcpu")
    for name in profiles:
        k = runnable(MEMORY_PROFILE=name, PARALLEL="ddp", **MULTI, **kw)
        assert validate(k) == [], (name, validate(k))


def test_full_ft_muon_runs_on_a_resident_profile():
    """Muon (whole-matrix steps after backward) validates on a resident profile; the
    streamed full-FT profile is rejected with the knobs that would fit."""
    muon = dict(ALGO="full", OPTIMIZER="muon", EXPORT_FORMATS=("merged",))

    def check(profile: str) -> None:
        assert validate(runnable(MEMORY_PROFILE=profile, **muon)) == []

    check("ckpt-resident")
    problems = validate(runnable(MEMORY_PROFILE="balanced", **muon))
    assert _has(problems, "trains blocks with AdamW", "ckpt-resident")
    must_fail(lambda: check("balanced"), "muon on swapped, fused blocks")


def test_8bit_is_an_optimizer_not_a_placement(monkeypatch):
    """B12: OPTIM_STATE places AdamW states only ('8bit' is no placement);
    OPTIMIZER='adamw8bit' without bitsandbytes is a knob problem; the internal
    placement builders are not OPTIMIZER names."""
    monkeypatch.setattr(optim_factory, "bnb", None)
    eight = runnable(MEMORY_OVERRIDES={"optim_state": "8bit", "fused_step": True})
    assert _has(validate(eight), "optim_state '8bit' not in")
    needle = "OPTIMIZER='adamw8bit' needs the bitsandbytes package"
    assert _has(validate(runnable(OPTIMIZER="adamw8bit")), needle)
    assert not _has(validate(runnable(OPTIMIZER="adamw8bit_kernel")), needle)
    for name in ("cpu_adamw", "host_state_adamw", "master_adamw"):
        assert _has(validate(runnable(OPTIMIZER=name)), "is the placement of")
    plans, _ = knobs_mod._plans(runnable(MEMORY_PROFILE="auto"))
    assert not [q for q in plans if q.optim_state not in ("gpu", "stream", "cpu")]
    assert not plan_problems(runnable(), plans[0])
    must_fail(
        lambda: _yes(_has(validate(runnable(OPTIMIZER="adamw")), "is the placement")),
        "plain adamw rejected",
    )


def _yes(flag: bool) -> None:
    assert flag
