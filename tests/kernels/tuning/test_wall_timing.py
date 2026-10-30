"""`wall_ms` times host launch work with the device work (a host-bound call costs its host
time), and `select(timing="wall")` resolves to it."""

import time

import pytest
import torch

from kohakumoss_kernels.tuning.timing import wall_ms
from kohakumoss_kernels.tuning.tuner import TIMINGS
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def _host_bound(ms: float):
    def fn():
        t = time.perf_counter()
        while (time.perf_counter() - t) * 1e3 < ms:
            pass

    return fn


def _check(measure) -> None:
    got = measure(_host_bound(2.0), torch.device("cpu"))
    assert 1.9 <= got <= 6.0, got


def test_wall_counts_host_time():
    _check(wall_ms)


def test_wall_counts_host_time_negative():
    must_fail(lambda: _check(lambda fn, device: 0.1), "host time not counted")


def test_registered():
    assert TIMINGS["wall"] is wall_ms
