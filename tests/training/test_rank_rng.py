"""Per-rank RNG under PARALLEL='ddp' (2 gloo processes on CPU): each rank draws its own
sigma / noise streams, checkpoints keep every rank's RNG, and resume restores each
rank's own state bit-exactly."""

import shutil
import sys

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.training.entry import run, stream_rank
from tests._util.negative import must_fail
from tests.training.test_entry_smoke import smoke_knobs

pytestmark = [pytest.mark.cpu, pytest.mark.slow]
DDP = dict(PARALLEL="ddp", DEVICES=(0, 1), DIST_BACKEND="gloo")


def _ddp_run(out, **kw) -> None:
    if sys.platform == "win32":
        pytest.skip("multi-process training is Linux-only")
    if torch.cuda.is_available():
        pytest.skip("CPU gloo test (hide the GPUs)")
    run(smoke_knobs(out, **DDP, SAVE_EVERY=2, KEEP_LAST=0, **kw))


def _ckpt(out, step: int):
    return out / "smoke" / "checkpoints" / f"step_{step:08d}"


def _rng(path, rank: int) -> dict:
    return torch.load(path / f"rng.{rank}.pt", weights_only=False)


def test_stream_rank_follows_data_rank():
    assert [stream_rank("ddp", r) for r in (0, 1)] == [0, 1]
    assert [stream_rank("fsdp", r) for r in (0, 1)] == [0, 1]
    assert [stream_rank("pipeline", r) for r in (0, 1)] == [0, 0]
    assert stream_rank("threaded", 0) == 0 == stream_rank("none", 0)


def _assert_distinct(a: dict, b: dict) -> None:
    for gen in ("sigma_gen", "noise_gen"):
        assert not torch.equal(a["step"][gen], b["step"][gen]), gen
    assert not torch.equal(a["torch"], b["torch"])


def test_ranks_draw_distinct_streams_and_save_their_own(tmp_path):
    _ddp_run(tmp_path, STEPS=2)
    ck = _ckpt(tmp_path, 2)
    r0, r1 = _rng(ck, 0), _rng(ck, 1)
    _assert_distinct(r0, r1)
    must_fail(lambda: _assert_distinct(r0, r0), "both ranks on one stream")


def _final(out) -> dict:
    return load_file(str(_ckpt(out, 4) / "trainable.safetensors"))


def _assert_same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_ddp_resume_restores_each_ranks_rng(tmp_path):
    full = tmp_path / "full"
    _ddp_run(full, STEPS=4)
    resumed = tmp_path / "resumed"
    _ddp_run(resumed, STEPS=4, RESUME=str(_ckpt(full, 2)))
    _assert_same(_final(full), _final(resumed))

    shared = tmp_path / "shared"
    source = tmp_path / "rank0_everywhere"
    shutil.copytree(_ckpt(full, 2), source)
    shutil.copy(source / "rng.0.pt", source / "rng.1.pt")
    _ddp_run(shared, STEPS=4, RESUME=str(source))
    must_fail(
        lambda: _assert_same(_final(full), _final(shared)),
        "rank 0's RNG restored on rank 1",
    )
