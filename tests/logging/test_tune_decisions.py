"""Kernel decision logging: a queued (provisional) binding is no decision, a problem is
recorded once whatever call pattern reaches it, a batch is one INFO summary table, and
after step 1 a newly measured problem is exactly one INFO line with its key and
timings."""

import logging
from itertools import cycle

import pytest
import torch

from kohakumoss_kernels.backends.compile import choice
from kohakumoss_kernels.tuning import decisions, tuner
from tests._util.logs import collecting
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
MS = {"impl=kernel": 0.21, "impl=compile": 0.17}


def _kernel(x):
    return x * 2.0


def _compiled(x):
    return x + x


@pytest.fixture
def op(monkeypatch, tmp_path):
    """A kernel-or-compile op with plain-python candidates, a fresh table directory,
    fixed timings."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    order = cycle(MS.values())
    spy = {k: (lambda fn, device: next(order)) for k in tuner.TIMINGS}
    monkeypatch.setattr(tuner, "TIMINGS", spy)
    choice.reset()
    tuner.set_mode("table")
    op = choice.kernel_or_compile("test", "situ_glu", _kernel, choice.shape_key)
    op.impls["compile"] = _compiled
    yield op
    choice.reset()


def _info(logs) -> list[str]:
    return [r.getMessage() for r in logs.records if r.levelno >= logging.INFO]


def _no_contradiction(lines: list[str], name: str) -> None:
    """At most one INFO record names the kernel, and none is a per-source line."""
    mine = [m for m in lines if name in m]
    assert len(mine) <= 1, mine
    assert not any(" from the " in m or "from a measurement" in m for m in mine), mine


def _row(lines: list[str], name: str) -> list[str]:
    table = next(m for m in lines if name in m)
    row = next(ln for ln in table.splitlines() if ln.strip().startswith(name))
    return [c.strip() for c in row.split("|")]


def test_queued_measured_second_pattern(op):
    with collecting(logging.DEBUG) as logs:
        op(torch.randn(256, 512))
        assert op.pending
        queued = [r for r in logs.records if r.getMessage().startswith("queued")]
        assert queued and all(r.levelno == logging.DEBUG for r in queued)
        assert choice.retune_pending() == 1
        op(torch.randn(200, 512))
    lines = _info(logs)
    _no_contradiction(lines, op.name)
    # columns: from table, measured, inherited, default
    assert _row(lines, op.name) == [op.name, "0", "1", "0", "0", "compile", "1"], lines
    debug = [r.getMessage() for r in logs.records if r.levelno == logging.DEBUG]
    assert any("kernel 0.21 ms, compile 0.17 ms" in m for m in debug), debug
    old = [
        f"{op.name}: config impl=kernel from the default",
        f"{op.name}: config impl=compile from a measurement",
        f"{op.name}: config impl=compile from the tuning table",
    ]
    must_fail(lambda: _no_contradiction(old, op.name), "the per-source lines")


def _one_line_with(lines: list[str], key: str) -> None:
    hits = [m for m in lines if key in m]
    assert len(hits) == 1 and "\n" not in hits[0], hits
    assert "kernel 0.21 ms, compile 0.17 ms" in hits[0] and "measured" in hits[0]


def test_mid_run_new_problem_is_one_line(op):
    op(torch.randn(256, 512))
    choice.retune_pending()
    decisions.go_live()
    x = torch.randn(1024, 512)
    key = op.problem([x])
    with collecting(logging.INFO) as logs:
        op(x)
        choice.retune_pending()
    _one_line_with(_info(logs), key)
    decisions.clear()
    y = torch.randn(4096, 512)
    with collecting(logging.INFO) as late:
        op(y)
        choice.retune_pending()
    must_fail(lambda: _one_line_with(_info(late), op.problem([y])), "batched, not live")
