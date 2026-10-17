"""The fused-vs-eager bar fails exactly the results that are clearly worse in both the
absolute and the relative sense."""

import pytest
import torch
from _fused_bar import WORSE, check, explodes

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
N = 65536


def _gen():
    return torch.Generator().manual_seed(0)


def _mixed(big_frac: float, big: float, small: float):
    """A reference with `big_frac` of its elements at magnitude `big`, the rest `small`,
    and a boolean mask of the big ones."""
    g = _gen()
    is_big = torch.rand(N, generator=g) < big_frac
    mag = torch.where(is_big, torch.tensor(big), torch.tensor(small))
    return torch.randn(N, generator=g, dtype=torch.float64) * mag, is_big


def _noisy(ref, per_elem_rel, seed):
    g = torch.Generator().manual_seed(seed)
    return ref + ref * per_elem_rel * torch.randn(N, generator=g, dtype=torch.float64)


def _scaled(ref, rel, mask, factor, seed):
    """Per-element relative noise `rel`, `factor` x on the elements in `mask`."""
    k = torch.where(mask, torch.tensor(factor), torch.tensor(1.0)).double()
    return _noisy(ref, rel * k, seed)


def test_same_precision_and_mildly_worse_pass():
    ref, _ = _mixed(0.5, 1.0, 1.0)
    eager = _noisy(ref, 1e-3, 1)
    assert not explodes(_noisy(ref, 1e-3, 2), eager, ref)
    assert not explodes(_noisy(ref, 1.5e-3, 2), eager, ref)


def test_a_scale_error_fails():
    ref, _ = _mixed(0.5, 1.0, 1.0)
    eager = _noisy(ref, 1e-3, 1)
    assert explodes(eager * (1 + 1e-2), eager, ref)
    must_fail(lambda: check(eager * (1 + 1e-2), eager, ref, "scaled"), "1% scale")


def test_worse_only_in_absolute_terms_passes():
    ref, big = _mixed(0.001, 100.0, 1.0)
    eager = _noisy(ref, 1e-3, 1)
    fused = _scaled(ref, 1e-3, big, 10.0, 2)
    assert not explodes(fused, eager, ref)
    assert explodes(fused, eager, ref, worse=1.0)


def test_worse_only_in_relative_terms_passes():
    ref, big = _mixed(0.5, 1.0, 0.05)
    eager = _noisy(ref, 1e-3, 1)
    fused = _scaled(ref, 1e-3, ~big, 10.0, 2)
    assert not explodes(fused, eager, ref)
    assert explodes(fused, eager, ref, worse=1.0)


def test_worse_in_both_fails():
    ref, _ = _mixed(0.5, 1.0, 0.05)
    eager = _noisy(ref, 1e-3, 1)
    fused = _noisy(ref, 1e-3 * 3 * WORSE, 2)
    assert explodes(fused, eager, ref)
