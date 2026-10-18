"""Kernel timeline breakdown on constructed chrome-trace events: unions, the NCCL time
hidden under compute, idle time, and the device filter."""

import pytest

from kohakumoss.parallel.telemetry.timeline import breakdown, overlap, union
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _ev(name, ts, dur, device=0, cat="kernel"):
    args = {"device": device}
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "args": args}


EVENTS = [
    _ev("gemm", 0, 10),
    _ev("gemm", 20, 10),
    _ev("softmax", 2, 3),
    _ev("ncclDevKernel_AllReduce_Sum_f32", 5, 20),
    _ev("Memcpy HtoD", 30, 2, cat="gpu_memcpy"),
    _ev("gemm", 0, 40, device=1),
    {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 40},
]


def test_union_and_overlap():
    assert union([(5, 8), (0, 3), (2, 4), (8, 9)]) == [(0, 4), (5, 9)]
    assert overlap([(0, 10), (20, 30)], [(5, 25)]) == 10


def check_breakdown(out: dict) -> None:
    """2 steps over 40 us: compute 20, NCCL 20 of which 10 under compute, copy 2, idle
    40 - 32 (busy [0, 32])."""
    want = {
        "trace_compute_ms": 10e-3,
        "trace_comm_ms": 10e-3,
        "trace_comm_exposed_ms": 5e-3,
        "trace_copy_ms": 1e-3,
        "trace_idle_ms": 4e-3,
        "trace_wall_ms": 20e-3,
    }
    for key, value in want.items():
        assert abs(out[key] - value) < 1e-12, (key, out[key], value)
    assert abs(out["trace_busy_frac"] - 0.8) < 1e-12


def test_breakdown_of_one_device():
    out = breakdown(EVENTS, 0, wall_us=40.0, steps=2)
    check_breakdown(out)
    assert out["trace_kernels_per_step"] == 2.5
    every_comm_exposed = {**out, "trace_comm_exposed_ms": out["trace_comm_ms"]}
    must_fail(lambda: check_breakdown(every_comm_exposed), "overlapped NCCL as exposed")
    other = breakdown(EVENTS, 1, wall_us=40.0, steps=2)
    must_fail(lambda: check_breakdown(other), "another device's kernels")
