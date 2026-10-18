"""Optimizer step choice (kohakumoss_kernels.optim.choice): one timed choice per problem
key between the kernel and the reference step, on scratch copies; then only lookups."""

import time

import pytest
import torch

from kohakumoss_kernels.optim import looped_multi, reference
from kohakumoss_kernels.optim.choice import KernelOrReference, problem_key
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
HP = (1e-3, 0.9, 0.95, 1e-8, 0.01)


@pytest.fixture
def timed(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    tuner.set_mode("table")
    yield
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()


def _slow(*args):
    time.sleep(0.002)
    reference.adamw_step(*args)


def _state(n: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    p = torch.randn(n, generator=gen)
    g = 0.1 * torch.randn(n, generator=gen)
    m = 0.01 * torch.randn(n, generator=gen)
    r = 0.01 * torch.rand(n, generator=gen)
    return p, g, m, r


def check_binds_fastest(kernel, fast_name: str):
    op = KernelOrReference("test", "adamw_step", kernel, reference.adamw_step)
    p, g, m, r = _state(4096)
    before = [t.clone() for t in (p, g, m, r)]
    op(p, g, m, r, None, *HP, 1, "nearest", 0)
    key = problem_key((p, g, m, r, None, *HP, 1, "nearest", 0), {})
    bound = op.bound[(p.device, key)]
    assert bound is op.impls[fast_name], f"bound {bound} instead of {fast_name}"
    ref = [t.clone() for t in before]
    reference.adamw_step(*ref, None, *HP, 1, "nearest", 0)
    for got, want in zip((p, m, r), (ref[0], ref[2], ref[3])):
        assert torch.equal(got, want), "the scratch timing run leaked into the step"


def test_slow_kernel_loses_to_reference(timed):
    check_binds_fastest(_slow, "torch")
    must_fail(lambda: check_binds_fastest(_slow, "kernel"), "the slower choice bound")


def test_bound_key_is_reused(timed):
    op = KernelOrReference("test", "adamw_step", _slow, reference.adamw_step)
    p, g, m, r = _state(4096)
    op(p, g, m, r, None, *HP, 1, "nearest", 0)
    timed_before = dict(tuner.TIMED)
    op(p, g, m, r, None, *HP, 2, "nearest", 0)
    assert tuner.TIMED == timed_before, "a bound problem was timed again"


def test_problem_key_separates_dtype_mode_and_size():
    a = _state(4096)
    b = _state(70000)
    base = problem_key((*a, None, *HP, 1, "nearest", 0), {})
    assert base != problem_key((*b, None, *HP, 1, "nearest", 0), {})
    assert base != problem_key((*a, None, *HP, 1, "kahan", 0), {})
    half = tuple(t.half() for t in a)
    assert base != problem_key((*half, None, *HP, 1, "nearest", 0), {})
    assert base == problem_key((*a, None, *HP, 7, "nearest", 3), {})


def test_multi_choice_matches_reference(timed):
    ref_multi = looped_multi(reference.adamw_step)
    op = KernelOrReference("test", "adamw_step_multi", ref_multi, ref_multi)
    states = [_state(n, i) for i, n in enumerate((7, 1024, 3000))]
    copies = [[t.clone() for t in s] for s in states]
    cols, rcols = list(zip(*states)), list(zip(*copies))
    op(*[list(c) for c in cols], None, *HP, 1, "nearest", 0)
    ref_multi(*[list(c) for c in rcols], None, *HP, 1, "nearest", 0)
    for s, c in zip(states, copies):
        assert all(torch.equal(x, y) for x, y in zip(s, c))
