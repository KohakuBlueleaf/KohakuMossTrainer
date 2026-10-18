"""Real training runs: the train script's entry on the real weights and a slice of the
KOHAKUMOSS_DATASET dataset, 3 steps, per memory profile x precision x algo. Each run must keep
every loss finite, take no non-finite step after the loss-scale warmup (an fp16 run
may overflow its first FP16_WARMUP steps at the initial scale while the scale backs
off; a bf16 run none), write one metric row per step with the
device-use peak, and log one step line per step. The profile runs take PROFILE_TOKENS
per batch, a budget at which `speed` (every activation, the VAE and the text encoder
resident) fits a 24 GB card; on a smaller card `speed` may run out of memory
(skipped); every other profile must run. Across profiles the first step's loss
and gradient norm agree with the most resident profile that ran (same data, seed,
weights, and the eager VAE encode in every profile, since `compile="auto"` compiles a
resident VAE only): offload, streaming and checkpointing change where tensors live,
never the math, but a resident and a streamed weight take different kernel paths, so
across profiles the bound is the larger of NOISE times the reference's measured
run-to-run noise and PATH_ULPS times the compute dtype's rounding step (EPS, relative).
A resume reruns one path: there the floor is REL_FLOOR relative."""

import gc
import json
import math
from pathlib import Path

import pytest
import torch

from kohakumoss.training.entry import run
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.negative import must_fail
from tests.integration.conftest import fp16_fix_knobs, real_knobs

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]

PROFILES = ("speed", "balanced", "minimal")
MAY_NOT_FIT = {"speed"}
ALGOS = {
    "lora": {"ALGO": "lora", "DIM": 8, "ALPHA": 8.0, "LR": 5e-4},
    "lokr": {
        "ALGO": "lokr",
        "DIM": 10000,
        "ALPHA": 1.0,
        "ALGO_KWARGS": {"factor": 16, "full_matrix": True},
        "LR": 2e-3,
    },
}
STEPS = 3
FP16_WARMUP = 1
PROFILE_TOKENS = 2048
NOISE = 4.0  # times the measured run-to-run difference
REL_FLOOR = 1e-4  # smallest noise floor, relative to the reference value
EPS = {"bf16": 2.0**-8, "fp16": 2.0**-11}  # the compute dtype's relative rounding step
PATH_ULPS = 2.0  # cross-path floor in rounding steps
DIVERGED = 1e-2  # a relative step-1 loss gap that is a real divergence


def _precision(name: str, weights_root: Path, fix_file) -> dict:
    match name:
        case "bf16":
            return {"PRECISION": "bf16"}
        case "fp16":
            if fix_file is None:
                pytest.skip("no fp16-fix file")
            return fp16_fix_knobs(weights_root, fix_file)


def _rows(out: Path) -> list[dict]:
    text = (out / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def warmup_of(precision: str) -> int:
    """Steps that may be non-finite at the start of a run: FP16_WARMUP for fp16, 0 else."""
    return FP16_WARMUP if precision == "fp16" else 0


def check_run(rows: list[dict], warmup: int = 0) -> None:
    """Every step logged, every loss finite, no non-finite step after the first `warmup`
    steps, at least one step applied, the device-use and allocator peaks recorded."""
    assert [r["step"] for r in rows] == list(range(1, STEPS + 1)), rows
    assert all(math.isfinite(r["loss"]) for r in rows), [r["loss"] for r in rows]
    bad = [r.get("nonfinite_steps", 0) for r in rows]
    settled = bad[warmup - 1] if warmup else 0
    assert all(b == settled for b in bad[warmup:]), (warmup, bad)
    assert any(math.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 for r in rows)
    for r in rows:
        assert r["vram_device"] > 0 and r["vram_peak_alloc"] > 0, r


def _gap(a: float, b: float) -> float:
    """|a - b|, 0 when either run gave a non-finite value (the floor applies)."""
    diff = abs(a - b)
    return diff if math.isfinite(diff) else 0.0


def bound(noise: float, ref: float, rel: float = REL_FLOOR) -> float:
    """The agreement bound of a check whose two reference runs differ by `noise`, at
    least `rel` relative to `ref`."""
    return max(NOISE * noise, NOISE * REL_FLOOR * abs(ref), rel * abs(ref))


def _close(a: float, b: float, noise: float, rel: float = REL_FLOOR) -> bool:
    return abs(a - b) <= bound(noise, b, rel)


def path_floor(precision: str) -> float:
    """The relative cross-path floor: PATH_ULPS rounding steps of the compute dtype."""
    return PATH_ULPS * EPS[precision]


def check_agree(first: dict, ref: tuple, noise: tuple, rel: float) -> None:
    """Each profile's step-1 (loss, grad norm) agrees with the resident profile's within
    the bound of its measured run-to-run noise and the cross-path floor `rel`."""
    for profile, (loss, gnorm) in first.items():
        assert _close(loss, ref[0], noise[0], rel), (profile, loss, ref, noise)
        if math.isfinite(gnorm) and math.isfinite(ref[1]):
            assert _close(gnorm, ref[1], noise[1], rel), (profile, gnorm, ref, noise)


def _profile_knobs(weights_root, dataset, out, profile, algo, prec) -> dict:
    return real_knobs(
        weights_root,
        dataset,
        out,
        MEMORY_PROFILE=profile,
        MAX_TOKENS=PROFILE_TOKENS,
        VAE_ENCODE={"compile": "off"},
        **ALGOS[algo],
        **prec,
    )


def run_fits(k: dict):
    """The run's final state, or None when it runs the device out of memory (the
    profile does not fit this card; its memory is released before returning)."""
    try:
        return run(k)
    except torch.OutOfMemoryError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return None


@pytest.mark.parametrize("algo", list(ALGOS))
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_real_runs_per_profile(
    precision, algo, weights_root, fix_file, dataset_slice, tmp_path
):
    prec = _precision(precision, weights_root, fix_file)
    first, rows = {}, []
    for profile in PROFILES:
        out = tmp_path / profile
        k = _profile_knobs(weights_root, dataset_slice, out, profile, algo, prec)
        state = run_fits(k)
        if state is None:
            assert profile in MAY_NOT_FIT, f"{profile} ran out of memory"
            continue
        assert state.step == STEPS, profile
        rows = _rows(out)
        check_run(rows, warmup_of(precision))
        first[profile] = (rows[0]["loss"], rows[0]["grad_norm"])
    ref_profile, ref = next(iter(first.items()))
    again = tmp_path / f"{ref_profile}-again"
    run(_profile_knobs(weights_root, dataset_slice, again, ref_profile, algo, prec))
    rerun = _rows(again)[0]
    noise = (_gap(rerun["loss"], ref[0]), _gap(rerun["grad_norm"], ref[1]))
    rel = path_floor(precision)
    check_agree(first, ref, noise, rel)
    nan_rows = [{**r, "loss": float("nan")} for r in rows]
    must_fail(lambda: check_run(nan_rows), "a run with non-finite losses")
    late = [{**r, "nonfinite_steps": min(r["step"], 2)} for r in rows]
    must_fail(
        lambda: check_run(late, warmup_of(precision)),
        "2 of 3 steps non-finite (one past the warmup)",
    )
    off = {
        p: (ref[0] + 2 * bound(noise[0], ref[0], rel), v[1]) for p, v in first.items()
    }
    must_fail(lambda: check_agree(off, ref, noise, rel), "a loss twice the bound off")
    far = {p: (ref[0] * (1 + DIVERGED), v[1]) for p, v in first.items()}
    must_fail(lambda: check_agree(far, ref, noise, rel), "a 1e-2 loss divergence")


def _checkpoint(out: Path, step: int) -> Path:
    return next((out / "run" / "checkpoints").glob(f"step_{step:08d}"))


def check_resume(straight: list[dict], resumed: list[dict], noise: dict) -> None:
    """The resumed run's steps 3-4 match the uninterrupted run's within the bound of
    each step's measured run-to-run noise."""
    tail = {r["step"]: r["loss"] for r in straight if r["step"] > 2}
    got = {r["step"]: r["loss"] for r in resumed if r["step"] > 2}
    assert got.keys() == tail.keys(), (got, tail)
    for step, loss in tail.items():
        assert _close(got[step], loss, noise[step]), (step, got[step], loss)


def test_real_resume_equals_uninterrupted(weights_root, dataset_slice, tmp_path):
    def knobs(out, **kw):
        return real_knobs(
            weights_root,
            dataset_slice,
            out,
            STEPS=4,
            SAVE_EVERY=2,
            MEMORY_PROFILE="balanced",
            **ALGOS["lora"],
            **kw,
        )

    run(knobs(tmp_path / "straight"))
    straight = _rows(tmp_path / "straight")
    run(knobs(tmp_path / "rerun"))
    rerun = {r["step"]: r["loss"] for r in _rows(tmp_path / "rerun")}
    noise = {r["step"]: _gap(r["loss"], rerun[r["step"]]) for r in straight}
    run(knobs(tmp_path / "resumed", RESUME=str(_checkpoint(tmp_path / "straight", 2))))
    check_resume(straight, _rows(tmp_path / "resumed"), noise)
    shifted = [
        {**r, "loss": r["loss"] + 2 * bound(noise[r["step"]], r["loss"])}
        for r in straight
    ]
    must_fail(
        lambda: check_resume(straight, shifted, noise), "a resume twice the bound off"
    )


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_real_run_on_an_fp8_compute_base(
    precision, weights_root, fix_file, dataset_slice, tmp_path
):
    """An sm89 LoKr setup: LoKr on an fp8_compute base (16-bit dgrad), balanced."""
    if not arch_of("cuda").has_fp8:
        pytest.skip("needs sm89+")
    k = real_knobs(
        weights_root,
        dataset_slice,
        tmp_path,
        MEMORY_PROFILE="balanced",
        BASE_PRECISION="fp8_compute",
        FP8_DGRAD="16bit",
        **ALGOS["lokr"],
        **_precision(precision, weights_root, fix_file),
    )
    assert run(k).step == STEPS
    check_run(_rows(tmp_path), warmup_of(precision))
