"""PARALLEL='fsdp' through the entry on two CUDA devices (Linux): the same gradient
norms, updates and full checkpoints as DDP on the same data, and an exact resume."""

import json

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.training.entry import run
from tests._util.negative import must_fail
from tests.training.test_entry_smoke import smoke_knobs
from tests.training.test_pipelines_gpu import EAGER

pytestmark = [pytest.mark.cuda, pytest.mark.multi_gpu, pytest.mark.slow]


def _run(tmp_path, name, kind, steps=2, **kw):
    knobs = smoke_knobs(
        tmp_path / name,
        PARALLEL=kind,
        DEVICES=(0, 1),
        DIST_BACKEND="nccl",
        KERNELS=EAGER,
        STEPS=steps,
        SAVE_EVERY=2,
        GRAD_ACC=2,
        SAMPLE_EVERY=0,
        EXPORT_FORMATS=(),
        **kw,
    )
    run(knobs)
    out = tmp_path / name / "smoke"
    ck = out / "checkpoints" / f"step_{steps:08d}"
    lines = (out / "metrics.jsonl").read_text().splitlines()
    return load_file(str(ck / "trainable.safetensors")), [json.loads(x) for x in lines]


def _close(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, k
        assert torch.allclose(a[k], b[k], rtol=1e-4, atol=1e-6), k


def _same_norms(pairs) -> None:
    for a, b in pairs:
        assert abs(a - b) <= 1e-4 * abs(a), pairs


@pytest.mark.parametrize("algo", ["lora", "full"])
def test_fsdp_matches_ddp(tmp_path, algo):
    ddp, ddp_m = _run(tmp_path, "ddp", "ddp", ALGO=algo)
    fsdp, fsdp_m = _run(tmp_path, "fsdp", "fsdp", ALGO=algo)
    _close(fsdp, ddp)
    pairs = [(a["grad_norm"], b["grad_norm"]) for a, b in zip(ddp_m, fsdp_m)]
    _same_norms(pairs)
    doubled = [(a, 2 * b) for a, b in pairs]
    must_fail(lambda: _same_norms(doubled), "FSDP gradients world x the mean")


def test_fsdp_resume_is_exact(tmp_path):
    straight, _ = _run(tmp_path, "straight", "fsdp", steps=4, ALGO="full")
    _run(tmp_path, "resumed", "fsdp", steps=2, ALGO="full")
    resumed, _ = _run(
        tmp_path, "resumed", "fsdp", steps=4, ALGO="full", RESUME="latest"
    )
    _close(resumed, straight)
