"""A tuning candidate that runs out of memory while timing is skipped for this process:
the problem binds the best config that ran (else the first), nothing is stored, and the
error never reaches the caller."""

import pytest
import torch

from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")


@pytest.fixture
def auto_tuning(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    tuner.set_mode("table")
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    yield
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()


def _launch(oom: set[int]):
    def launch(config):
        def run():
            if config["a"] in oom:
                raise torch.OutOfMemoryError("CUDA out of memory (test)")

        return run

    return launch


def check_oom_skipped(kernel: str, oom: set[int], want: int) -> None:
    configs = [{"a": 1}, {"a": 2}, {"a": 3}]
    try:
        got = tuner.select(kernel, CPU, "k", lambda: configs, _launch(oom))
    except torch.OutOfMemoryError as err:
        raise AssertionError(f"tuning OOM reached the caller: {err}") from err
    assert got == {"a": want}, got
    assert tuner.decided(kernel, CPU, "k")
    assert not tuner.stored(kernel, CPU, "k"), "an OOM-affected choice was stored"


def test_oom_candidate_is_skipped(auto_tuning):
    check_oom_skipped("test_oom_some", {1, 3}, 2)


def test_every_candidate_oom_keeps_the_default(auto_tuning):
    check_oom_skipped("test_oom_all", {1, 2, 3}, 1)


def test_negative_oom_propagates(auto_tuning, monkeypatch):
    monkeypatch.setattr(tuner, "OOM_SKIPPED", ())
    must_fail(lambda: check_oom_skipped("test_oom_raise", {1, 3}, 2), "OOM stored")
