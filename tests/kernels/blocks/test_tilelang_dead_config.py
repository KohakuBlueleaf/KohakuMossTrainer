"""TileLang fused linears: a fused config that fails to compile is dropped and the
problem runs the next candidate, under TUNE=off as well."""

import pytest

from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytest.importorskip("tilelang")
fused_linear = pytest.importorskip(
    "kohakumoss_kernels.backends.tilelang.blocks.fused_linear"
)
blockops = pytest.importorskip("kohakumoss_kernels.backends.tilelang.blocks._blockops")

pytestmark = pytest.mark.cpu
CFGS = [
    {"path": "fused", "bm": 64, "bn": 64, "bk": 32, "threads": 128, "stages": 1},
    {"path": "split"},
]


def _fused(cfg, x):
    raise blockops.CompileError("does not lower")


def _split(cfg, x):
    return ("split", x)


PATHS = {"fused": _fused, "split": _split}


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(blockops, "_DEAD", set())
    monkeypatch.setattr(fused_linear.FUSED, "bound", {})
    tuner.reset()
    yield
    tuner.reset()


def _no_retry(name, device, key, cfgs, paths, args):
    return fused_linear.FUSED._bound(name, device, key, cfgs, paths, args)(*args)


def _check_falls_back(run, name: str) -> None:
    for _ in range(2):
        try:
            got = run(name, "cpu", "k", CFGS, PATHS, (3,))
        except blockops.CompileError as err:
            raise AssertionError(f"re-raised: {err}") from err
        assert got == ("split", 3)


def test_failing_default_falls_back_to_split(fresh):
    _check_falls_back(fused_linear.FUSED.run, "t_dead_a")
    assert fused_linear.FUSED.bound[("t_dead_a", "cpu", "k")].args[0] == CFGS[1]
    must_fail(
        lambda: _check_falls_back(_no_retry, "t_dead_b"),
        "fused failure re-raised on every call",
    )
