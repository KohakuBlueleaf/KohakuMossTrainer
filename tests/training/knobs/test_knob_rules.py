"""Cross-knob rules: combinations that fail or misbehave after the build are reported by
`validate`, together with every other problem."""

import sys

import pytest

from kohakumoss.training import knobs
from kohakumoss.training.knobs import validate
from kohakumoss_kernels.hardware.arch import Arch
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu


def _has(problems: list[str], *needles: str) -> None:
    joined = "\n".join(problems)
    for needle in needles:
        assert needle in joined, (needle, problems)


def _clean(k: dict) -> None:
    assert validate(k) == []


def test_cache_dir_resolution_is_checked():
    offline = dict(LATENT_CACHE="offline", TEXT_CACHE="offline")
    _has(validate(runnable(**offline)), "CACHE_DIR")
    with_path = runnable(**offline, DATASETS=[{"source": "kohya", "path": "x/data"}])
    assert not any("CACHE_DIR" in p for p in validate(with_path))
    assert not any(
        "CACHE_DIR" in p for p in validate(runnable(**offline, CACHE_DIR="c"))
    )


def test_masked_mse_without_a_mask_source():
    _has(validate(runnable(LOSS="masked_mse")), "LOSS='masked_mse'")
    _clean(runnable(LOSS="mse"))


def _warmup_fix(problems: list[str]) -> None:
    _has(
        problems,
        "warmup=100 >= its end 30 (STEPS)",
        "below 30",
        "raise STEPS above 100",
    )


def test_warmup_past_steps_names_the_fix_and_keeps_the_schedule():
    sched = {"mode": "cosine", "warmup": 100, "end": -1}
    k = runnable(STEPS=30, SCHEDULE=sched)
    _warmup_fix(validate(k))
    assert k["SCHEDULE"] == {"mode": "cosine", "warmup": 100, "end": -1}
    old = ["SCHEDULE warmup 100 >= end 30 (STEPS=30)"]
    must_fail(lambda: _warmup_fix(old), "message without the fix")
    _clean(runnable(STEPS=30, SCHEDULE={**sched, "warmup": 3}))


@pytest.mark.parametrize("variant", ["base", "sft"])
def test_fp16_on_a_real_variant_needs_a_fix(variant):
    k = runnable(VARIANT=variant, PRECISION="fp16", FP16_FIX="none", FP16_FIX_KWARGS={})
    _has(validate(k), "FP16_FIX")
    for fix, kwargs in (("online", {"c": 1 / 16}), ("fp32_residual", {})):
        fixed = runnable(
            VARIANT=variant, PRECISION="fp16", FP16_FIX=fix, FP16_FIX_KWARGS=kwargs
        )
        assert not any("FP16_FIX" in p for p in validate(fixed))
    assert not any("FP16_FIX" in p for p in validate(runnable(PRECISION="fp16")))


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")


def test_muon_with_fsdp(linux):
    k = runnable(OPTIMIZER="muon", PARALLEL="fsdp", DEVICES=(0, 1))
    _has(validate(k), "OPTIMIZER='muon'")
    assert not any(
        "OPTIMIZER='muon'" in p
        for p in validate(runnable(OPTIMIZER="muon", PARALLEL="ddp", DEVICES=(0, 1)))
    )


def test_threaded_schedule_is_checked(linux):
    base = dict(PARALLEL="threaded", DEVICES=(0, 1))
    _has(validate(runnable(**base, PIPELINE_SCHEDULE="1f1b")), "PIPELINE_SCHEDULE")
    for schedule in ("gpipe", "async"):
        problems = validate(runnable(**base, PIPELINE_SCHEDULE=schedule))
        assert not any("PIPELINE_SCHEDULE" in p for p in problems)


def _arch(name: str, bf16: bool) -> Arch:
    return Arch(name, 75 if not bf16 else 86, "cuda", bf16, False, True)


@pytest.fixture
def mixed_host(monkeypatch):
    """cuda:0 an sm86, every other index an sm75 (no bf16)."""
    monkeypatch.setattr(knobs.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(knobs.torch.cuda, "device_count", lambda: 4)

    def arch_of(device):
        index = knobs.torch.device(device).index or 0
        return _arch("sm86", True) if index == 0 else _arch("sm75", False)

    monkeypatch.setattr(knobs, "arch_of", arch_of)
    monkeypatch.setattr(knobs, "default_device_type", lambda: "cuda")


def test_fp32_is_not_a_compute_precision(mixed_host, linux, monkeypatch):
    _has(validate(runnable(PRECISION="fp32")), "PRECISION='fp32'")
    assert not any("fp32" in p for p in validate(runnable(PRECISION="bf16")))
    monkeypatch.setattr(knobs, "default_device_type", lambda: "cpu")
    _has(validate(runnable(PRECISION="fp32")), "PRECISION='fp32'")


def test_auto_precision_resolves_per_device_set(mixed_host, linux, monkeypatch):
    assert knobs.resolve_precision(runnable(DEVICES=(0,))) == "bf16"
    assert knobs.resolve_precision(runnable(PARALLEL="ddp", DEVICES=(0, 2))) == "fp16"
    assert validate(runnable(PARALLEL="ddp", DEVICES=(0, 2))) == []
    assert knobs.resolve_precision(runnable(PRECISION="fp16", DEVICES=(0,))) == "fp16"
    monkeypatch.setattr(knobs, "default_device_type", lambda: "cpu")
    assert knobs.resolve_precision(runnable()) == "bf16"


def test_bf16_rule_checks_every_device(mixed_host, linux):
    _has(validate(runnable(PRECISION="bf16", DEVICES=(2,))), "PRECISION='bf16'")
    k = runnable(PRECISION="bf16", PARALLEL="ddp", DEVICES=(0, 2))
    _has(validate(k), "PRECISION='bf16'")
    assert not any("bf16" in p for p in validate(runnable(PRECISION="bf16")))
    must_fail(
        lambda: _has(validate(runnable(PRECISION="bf16", DEVICES=(0,))), "bf16"),
        "an sm86-only device set reported",
    )
