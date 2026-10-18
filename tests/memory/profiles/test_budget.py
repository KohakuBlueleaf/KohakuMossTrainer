"""The `auto` planner's budget: free memory on XPU as on CUDA (not the card's total), and
a measured free-memory budget does not count the context a second time."""

import pytest
import torch

from kohakumoss.memory import planner
from kohakumoss.memory.estimate import FRAGMENTATION, Estimate
from kohakumoss.offload import accel
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
GIB = 2**30


def _equal(a, b) -> None:
    assert a == b, (a, b)


def test_xpu_budget_is_free_memory(monkeypatch):
    props = type("Props", (), {"total_memory": 16 * GIB})()
    monkeypatch.setattr(
        torch.xpu, "mem_get_info", lambda d: (6 * GIB, 16 * GIB), raising=False
    )
    monkeypatch.setattr(
        torch.xpu, "get_device_properties", lambda d: props, raising=False
    )
    budget = planner.device_budget("xpu")
    assert budget == int(6 * GIB * accel.HEADROOM)
    total_based = int(props.total_memory * accel.HEADROOM)
    must_fail(lambda: _equal(total_based, budget), "the card's total taken as free")


@pytest.fixture
def fixed(monkeypatch):
    """Every profile estimates 10 GiB of which 1 GiB is the context; free is 9.5 GiB."""
    est = Estimate(parts={"context": GIB, "weights": int(9 * GIB / FRAGMENTATION)})
    monkeypatch.setattr(planner, "estimate", lambda work, plan: est)
    monkeypatch.setattr(planner, "device_budget", lambda device: int(9.5 * GIB))
    return est


def test_measured_budget_counts_the_context_once(fixed):
    plan, report = planner.auto_plan(None, "cpu")
    assert report.chosen == planner.ORDER[0], report.rows
    _, given = planner.auto_plan(None, "cpu", budget=int(9.5 * GIB), force=True)
    assert given.chosen.endswith("(nothing fits, forced)"), given.rows
    must_fail(
        lambda: _equal(given.chosen, planner.ORDER[0]), "the context counted twice"
    )


def _refuses() -> None:
    try:
        planner.auto_plan(None, "cpu", budget=int(9.5 * GIB))
    except planner.NothingFits as err:
        assert "MEMORY_AUTO_FORCE" in str(err) and err.report.rows
        return
    raise AssertionError("auto ran a profile that does not fit")


def test_nothing_fits_is_an_error_unless_forced(fixed, monkeypatch):
    _refuses()
    real = planner.auto_plan
    monkeypatch.setattr(
        planner, "auto_plan", lambda *a, **kw: real(*a, **{**kw, "force": True})
    )
    must_fail(_refuses, "nothing fits, ran anyway")
