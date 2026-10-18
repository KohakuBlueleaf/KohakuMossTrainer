"""`tuning.broadcast` replaces every rank's tables with rank 0's and drops every binding
built from a rank's own choices, so all ranks run the same kernels."""

import pytest
import torch.distributed as dist

from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail


@pytest.fixture
def one_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path / "cache"))
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{(tmp_path / 'store').as_posix()}",
        rank=0,
        world_size=1,
    )
    yield
    dist.destroy_process_group()
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()


def check_bindings_dropped() -> None:
    bound = {"kernel": "rank-local choice"}
    tuner.on_reset(bound.clear)
    try:
        tuner._CHOSEN[("cpu", "k", "key")] = {"cfg": 1}
        tuner.broadcast(0)
        assert not bound, "a binding cache survived the broadcast"
        assert not tuner._CHOSEN
    finally:
        tuner._RESETS.remove(bound.clear)


@pytest.mark.cpu
def test_broadcast_drops_bindings(one_rank):
    check_bindings_dropped()


@pytest.mark.cpu
def test_negative_broadcast_without_reset(one_rank, monkeypatch):
    monkeypatch.setattr(tuner, "reset", lambda: tuner._CHOSEN.clear())
    must_fail(check_bindings_dropped, "reset hooks not run")
