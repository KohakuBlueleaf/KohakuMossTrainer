"""Multi-GPU training on the real model (two CUDA devices, Linux), 3 steps each: DDP and
FSDP2 agree on step 1 (same data, same per-rank noise); DDP under a streaming, an
in-backward-step and a streaming + offload + in-backward-step profile keeps the ranks on
one summed gradient and follows the resident DDP run; the process-group pipeline and
the threaded pipeline reproduce, step by step, the single-GPU run that takes the same
micro-batches as gradient accumulation; every rank writes its per-rank report and rank 0
the job summary."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from kohakumoss.training.entry import run
from kohakumoss_data.cache.build import build_text
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import build_encoder, data_config
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.negative import must_fail
from tests.integration.conftest import real_knobs

pytestmark = [
    pytest.mark.real_weights,
    pytest.mark.cuda,
    pytest.mark.multi_gpu,
    pytest.mark.slow,
]

STEPS = 3
SPAWNED = ("ddp", "fsdp", "pipeline")
AGREE = 0.05  # relative, DDP vs FSDP2 step 1 (different reduction layouts)
SAME = 0.02  # relative, a pipeline vs its gradient-accumulation reference, every step
LORA = {"ALGO": "lora", "DIM": 8, "ALPHA": 8.0, "LR": 5e-4}
STATIC = {
    "TEXT_CACHE": "offline",
    "TOKEN_BUDGET": "exact",
    "BATCHING": "tokens_ffd",
    "MAX_SAMPLES": 4,
}


@pytest.fixture(scope="module")
def two_gpus():
    if sys.platform == "win32" or torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices on Linux")


FP16 = {"PRECISION": "fp16", "FP16_FIX": "online", "FP16_FIX_KWARGS": {"c": 1 / 16}}


def _precision() -> dict:
    return {"PRECISION": "bf16"} if arch_of("cuda:0").has_bf16 else FP16


def _knobs(weights_root, dataset, out: Path, **kw) -> dict:
    base = dict(
        MEMORY_PROFILE="ckpt-resident",
        METRICS_EVERY=1,
        TRACE_STEPS=(),
        PIPELINE_SPLIT="even",
        **LORA,
        **_precision(),
    )
    return real_knobs(weights_root, dataset, out, **{**base, **kw})


@pytest.fixture(scope="module")
def text_cache(weights_root, dataset_slice, tmp_path_factory, two_gpus) -> Path:
    """The offline text cache of the slice (every caption variant, the real encoder)."""
    cache = tmp_path_factory.mktemp("text_cache")
    k = _knobs(weights_root, dataset_slice, cache, CACHE_DIR=str(cache))
    encode = build_encoder(
        "kohakumoss.encoders.text_encoder",
        {
            "path": str(weights_root / "base" / "text_encoder"),
            "device": "cuda:0",
            "plan": {"memory": "stream"},
        },
    )
    spec = getattr(encode, "encoder_spec", "")
    cfg = data_config(
        k, latent_cache="off", text_cache="offline", text_encoder_spec=spec
    )
    build_text(MossDataset(cfg), encode, 64)
    del encode
    torch.cuda.empty_cache()
    return cache


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _run(k: dict) -> dict:
    """Train; the rank-0 metric rows, the per-rank rows and the job summary. A spawned
    job returns no state (`check_report` checks its steps from the rows)."""
    state = run(k)
    assert state is None if k["PARALLEL"] in SPAWNED else state.step == STEPS, state
    out = Path(k["OUTPUT_DIR"]) / k["RUN_NAME"]
    files = sorted((out / "ranks").glob("rank*.jsonl"))
    ranks = {int(p.stem[4:]): _rows(p) for p in files}
    summary = json.loads((out / "ranks" / "summary.json").read_text(encoding="utf-8"))
    return {"rows": _rows(out / "metrics.jsonl"), "ranks": ranks, "summary": summary}


def check_report(res: dict, world: int, engine_keys: tuple[str, ...] = ()) -> None:
    """Every step logged with a finite loss; one per-rank file per rank with a row per
    step carrying throughput, VRAM peaks and the engine's breakdown; a job summary."""
    rows = res["rows"]
    assert [r["step"] for r in rows] == list(range(1, STEPS + 1)), rows
    assert all(math.isfinite(r["loss"]) for r in rows), rows
    assert sorted(res["ranks"]) == list(range(world)), res["ranks"].keys()
    for rank, rank_rows in res["ranks"].items():
        assert [r["step"] for r in rank_rows] == list(range(1, STEPS + 1)), rank
        for r in rank_rows:
            assert r["samples_per_s"] > 0 and r["vram_peak_alloc_gib"] > 0, r
            assert all(key in r for key in engine_keys), (engine_keys, r)
    job = res["summary"]["job"]
    assert job["world"] == world and job["samples_per_s"] > 0, job


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def check_same_losses(got: list[dict], ref: list[dict], tol: float, steps) -> None:
    """Losses (and step-1 gradient norms) agree with the reference at `steps`."""
    a = {r["step"]: r for r in got}
    b = {r["step"]: r for r in ref}
    for s in steps:
        assert _close(a[s]["loss"], b[s]["loss"], tol), (s, a[s]["loss"], b[s]["loss"])
    assert _close(a[1]["grad_norm"], b[1]["grad_norm"], tol), (a[1], b[1])


def _shifted(rows: list[dict]) -> list[dict]:
    return [{**r, "loss": r["loss"] * 1.5} for r in rows]


def test_ddp_and_fsdp_agree(weights_root, dataset_slice, tmp_path, two_gpus):
    def knobs(kind: str) -> dict:
        return _knobs(
            weights_root,
            dataset_slice,
            tmp_path / kind,
            PARALLEL=kind,
            DEVICES=(0, 1),
            DIST_BACKEND="nccl",
        )

    ddp = _run(knobs("ddp"))
    check_report(ddp, 2, ("ddp_comm_wait_ms", "ddp_reduce_mb"))
    fsdp = _run(knobs("fsdp"))
    check_report(fsdp, 2)
    check_same_losses(fsdp["rows"], ddp["rows"], AGREE, [1])
    must_fail(
        lambda: check_same_losses(_shifted(fsdp["rows"]), ddp["rows"], AGREE, [1]),
        "an FSDP2 run with a different loss",
    )
    broken = {**ddp, "ranks": {0: ddp["ranks"][0]}}
    must_fail(lambda: check_report(broken, 2), "a rank without its report")


DDP_PROFILES = (
    "ckpt-stream4",
    "ckpt-resident-optfused",
    "ckpt-stream2-actoff-optstream-gradoff",
)


def _ddp_knobs(weights_root, dataset, out: Path, profile: str) -> dict:
    """DDP over two GPUs without clipping (a fused step clips per parameter)."""
    return _knobs(
        weights_root,
        dataset,
        out,
        PARALLEL="ddp",
        DEVICES=(0, 1),
        DIST_BACKEND="nccl",
        GRAD_CLIP=0.0,
        MEMORY_PROFILE=profile,
    )


@pytest.fixture(scope="module")
def ddp_resident(weights_root, dataset_slice, tmp_path_factory, two_gpus) -> dict:
    """DDP on the resident profile: the reference of the memory-profile runs."""
    out = tmp_path_factory.mktemp("ddp_resident")
    res = _run(_ddp_knobs(weights_root, dataset_slice, out, "ckpt-resident"))
    check_report(res, 2, ("ddp_comm_wait_ms", "ddp_reduce_mb"))
    return res


def check_ranks_agree(res: dict, tol: float = 1e-4) -> None:
    """Every rank stepped on the same summed gradient: equal gradient norms per step."""
    norms = {
        rank: {r["step"]: r["grad_norm"] for r in rows}
        for rank, rows in res["ranks"].items()
    }
    for step, norm in norms[0].items():
        for rank, other in norms.items():
            assert _close(other[step], norm, tol), (step, rank, other[step], norm)


def _split_norms(res: dict) -> dict:
    rows = [{**r, "grad_norm": r["grad_norm"] * 0.7} for r in res["ranks"][1]]
    return {**res, "ranks": {**res["ranks"], 1: rows}}


@pytest.mark.parametrize("profile", DDP_PROFILES)
def test_ddp_memory_profiles(
    profile, weights_root, dataset_slice, ddp_resident, tmp_path
):
    """Streaming, the in-backward step and gradient offload under DDP: the ranks agree
    on every step's gradient, and the run follows the resident DDP run."""
    res = _run(_ddp_knobs(weights_root, dataset_slice, tmp_path, profile))
    check_report(res, 2, ("ddp_comm_wait_ms", "ddp_span_ms", "ddp_reduce_mb"))
    check_ranks_agree(res)
    must_fail(lambda: check_ranks_agree(_split_norms(res)), "ranks without the sum")
    steps, ref = range(1, STEPS + 1), ddp_resident["rows"]
    check_same_losses(res["rows"], ref, AGREE, steps)
    must_fail(
        lambda: check_same_losses(_shifted(res["rows"]), ref, AGREE, steps),
        "a DDP run whose loss differs from the resident one",
    )


@pytest.fixture(scope="module")
def accumulated(weights_root, dataset_slice, text_cache, tmp_path_factory) -> dict:
    """The single-GPU reference: GRAD_ACC=2 on the static micro-batch knobs."""
    out = tmp_path_factory.mktemp("acc2")
    k = _knobs(
        weights_root,
        dataset_slice,
        out,
        GRAD_ACC=2,
        CACHE_DIR=str(text_cache),
        **STATIC,
    )
    res = _run(k)
    check_report(res, 1)
    return res


@pytest.mark.parametrize("schedule", ["1f1b", "gpipe"])
def test_pipeline_matches_accumulation(
    schedule, weights_root, dataset_slice, text_cache, accumulated, tmp_path
):
    k = _knobs(
        weights_root,
        dataset_slice,
        tmp_path,
        PARALLEL="pipeline",
        DEVICES=(0, 1),
        DIST_BACKEND="nccl",
        MICROBATCHES=2,
        PIPELINE_SCHEDULE=schedule,
        CACHE_DIR=str(text_cache),
        **STATIC,
    )
    res = _run(k)
    check_report(res, 2, ("pipe_compute_ms", "pipe_bubble_frac", "pipe_pad_frac"))
    steps, ref = range(1, STEPS + 1), accumulated["rows"]
    check_same_losses(res["rows"], ref, SAME, steps)
    must_fail(
        lambda: check_same_losses(_shifted(res["rows"]), ref, SAME, steps),
        "a pipeline whose loss differs from the accumulated run",
    )


def test_threaded_matches_accumulation(
    weights_root, dataset_slice, text_cache, accumulated, tmp_path
):
    k = _knobs(
        weights_root,
        dataset_slice,
        tmp_path,
        PARALLEL="threaded",
        DEVICES=(0, 1),
        MICROBATCHES=2,
        PIPELINE_SCHEDULE="gpipe",
        CACHE_DIR=str(text_cache),
        **STATIC,
    )
    res = _run(k)
    check_report(res, 1)
    check_same_losses(res["rows"], accumulated["rows"], SAME, range(1, STEPS + 1))
