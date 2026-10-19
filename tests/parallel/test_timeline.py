"""Kernel timeline breakdown on constructed chrome-trace events: unions, the NCCL time
hidden under compute, idle time, the device filter, and the compute time per step
phase."""

import pytest

from kohakumoss.parallel.telemetry.timeline import (
    breakdown,
    overlap,
    phase_breakdown,
    union,
)
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


def _range(name, ts, dur, tid):
    return {
        "ph": "X",
        "cat": "user_annotation",
        "name": name,
        "ts": ts,
        "dur": dur,
        "tid": tid,
    }


def _launch(corr, ts, tid, cat="cuda_runtime"):
    args = {"correlation": corr}
    return {
        "ph": "X",
        "cat": cat,
        "name": "launch",
        "ts": ts,
        "dur": 1,
        "tid": tid,
        "args": args,
    }


def _kernel(name, ts, dur, corr, device=0):
    return {
        **_ev(name, ts, dur, device),
        "args": {"device": device, "correlation": corr},
    }


# thread 1: data, forward, backward [100, 300], optimizer; thread 2 (autograd)
# launches the backward kernels, with a recompute range [150, 200]
PHASE_EVENTS = [
    _range("kohakumoss.data", 0, 10, 1),
    _range("kohakumoss.forward", 10, 90, 1),
    _range("kohakumoss.backward", 100, 200, 1),
    _range("kohakumoss.recompute", 150, 50, 2),
    _range("kohakumoss.optimizer", 300, 20, 1),
    _range("other.range", 20, 5, 1),
    _launch(1, 5, 1),
    _launch(2, 20, 1, "cuda_driver"),
    _launch(3, 120, 2),
    _launch(4, 160, 2),
    _launch(5, 210, 2),
    _launch(6, 310, 1),
    _launch(7, 400, 1),
    _kernel("copy_kernel", 6, 2, 1),
    _kernel("gemm", 30, 40, 2),
    _kernel("gemm_bwd", 130, 20, 3),
    _kernel("gemm", 170, 10, 4),
    _kernel("gemm_bwd", 215, 30, 5),
    _kernel("adam", 312, 4, 6),
    _kernel("late", 402, 6, 7),
    _kernel("gemm", 30, 40, 2, device=1),
    _ev("ncclDevKernel_AllReduce_Sum_f32", 220, 50),
]


def check_phases(out: dict) -> None:
    """One step: data 2, forward 40, backward 20 + 30, recompute 10, optimizer 4, other
    6 (us; the NCCL kernel is not compute)."""
    want = {"data": 2, "forward": 40, "backward": 50, "recompute": 10, "optimizer": 4}
    for p, us in {**want, "other": 6}.items():
        got = out[f"trace_phase_{p}_ms"]
        assert abs(got - us * 1e-3) < 1e-12, (p, got, us)


def test_phase_breakdown():
    out, top = phase_breakdown(PHASE_EVENTS, 0, steps=1)
    check_phases(out)
    assert top["backward"][0] == {
        "kernel": "gemm_bwd",
        "ms": pytest.approx(50e-3),
        "calls": 2.0,
    }
    assert top["recompute"] == [
        {"kernel": "gemm", "ms": pytest.approx(10e-3), "calls": 1.0}
    ]
    no_recompute = [e for e in PHASE_EVENTS if e.get("name") != "kohakumoss.recompute"]
    must_fail(
        lambda: check_phases(phase_breakdown(no_recompute, 0, 1)[0]),
        "recompute kernels counted as backward",
    )
    must_fail(
        lambda: check_phases(phase_breakdown(PHASE_EVENTS, 1, 1)[0]),
        "another device's kernels",
    )
