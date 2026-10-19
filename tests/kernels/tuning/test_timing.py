"""Tuning never captures a CUDA graph: a capture in the training process breaks CUDA
calls of other threads (a DataLoader pin thread), so the tuner times with events; a
burst queued behind a device spin times device work, not host launch time."""

import time

import pytest
import torch

from kohakumoss_kernels.tuning import tuner
from kohakumoss_kernels.tuning.timing import best_ms, burst_ms
from tests._util.negative import must_fail


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    tuner.set_mode("table")
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    yield
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()


def check_no_capture(monkeypatch, device, **select_kwargs) -> None:
    def no_capture(*args, **kwargs):
        raise AssertionError("the tuner captured a CUDA graph")

    monkeypatch.setattr(torch.cuda, "graph", no_capture)
    x = torch.randn(1024, device=device)
    picked = tuner.select(
        "timing_probe",
        device,
        str(select_kwargs),
        lambda: [{"scale": 1.0}, {"scale": 2.0}],
        lambda c: lambda: x * c["scale"],
        **select_kwargs,
    )
    assert picked in ({"scale": 1.0}, {"scale": 2.0})


@pytest.mark.cuda
def test_tuner_times_without_graph_capture(device, fresh, monkeypatch):
    check_no_capture(monkeypatch, device)


@pytest.mark.cuda
def test_negative_graph_timing_captures(device, fresh, monkeypatch):
    must_fail(lambda: check_no_capture(monkeypatch, device, timing="graph"))


def check_device_time(ms: float) -> None:
    """A call of ~2 ms host time and a few us of device work times as device work."""
    assert ms < 0.5, f"{ms:.3f} ms per call: host launch time was counted"


@pytest.mark.cuda
def test_burst_times_device_work_not_host_launch(device):
    x = torch.randn(1 << 16, device=device)

    def host_bound():
        time.sleep(2e-3)
        x.mul_(1.0)

    check_device_time(burst_ms(host_bound, device, calls=6, iters=2))
    must_fail(lambda: check_device_time(best_ms(host_bound, device, iters=2)))
