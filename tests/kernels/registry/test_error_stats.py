"""The error-statistics check itself: regressions only, stats table written, worse-list
kept."""

import json

import pytest
import torch

from kohakumoss_kernels.error_stats import compare, worse
from tests._util.error_stats import assert_no_regression, regressions
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def case(n: int = 4096):
    g = torch.Generator().manual_seed(0)
    r = torch.randn(n, generator=g, dtype=torch.float64)
    return r, r.half()


def test_equal_stats_pass():
    r, e = case()
    assert not regressions(compare(e, e, r))
    assert_no_regression(e, e, r, "equal")


def test_non_finite_regresses():
    r, e = case()
    bad = e.clone()
    bad[5] = float("nan")
    assert regressions(compare(bad, e, r)) == ["non-finite"]
    must_fail(lambda: assert_no_regression(bad, e, r, "nan"), "NaN output")


def test_scale_error_regresses():
    r, e = case()
    scaled = (r * 1.1).half()
    assert regressions(compare(scaled, e, r))
    must_fail(lambda: assert_no_regression(scaled, e, r, "scale"), "10% scale error")


def test_one_bad_element_is_listed_but_passes():
    r, e = case()
    one = e.clone()
    one[7] = (r[7] * 1.5).half()
    table = compare(one, e, r)
    assert not regressions(table)
    assert "abs_max" in worse(table)
    assert_no_regression(one, e, r, "one element")


def _near_zero_case():
    r, _ = case()
    r[:20] = 1e-6
    e = r.half()
    k = e.clone()
    k[:20] = 1e-3
    return r, e, k


def test_rel_alone_from_near_zero_reference_passes():
    """Large rel error only where the reference is near zero, abs error and cosine at
    eager's: rel and abs are read together, so this is listed, not a regression."""
    r, e, k = _near_zero_case()
    table = compare(k, e, r)
    assert table["kernel"]["rel_mean"] > 10 * table["eager"]["rel_mean"]
    assert "rel_mean" in worse(table)
    assert not regressions(table)
    assert_no_regression(k, e, r, "rel only")


def test_abs_and_rel_together_regress():
    r, e, k = _near_zero_case()
    k = (k.double() * 1.1).half()
    assert {"abs_mean", "rel_mean"} <= set(regressions(compare(k, e, r)))
    must_fail(lambda: assert_no_regression(k, e, r, "abs and rel"), "abs + rel off")


def test_exact_eager_one_rounding_off_passes():
    """Eager exact (all zero error); the kernel one fp16 rounding off everywhere is not
    a regression: eager's zero counts as the dtype's rounding scale."""
    r = torch.linspace(1, 2, 1024, dtype=torch.float64).half().double()
    e = r.half()
    off = torch.nextafter(e, torch.full_like(e, 4.0))
    table = compare(off, e, r)
    assert table["eager"]["abs_median"] == 0
    assert not regressions(table)


def test_exact_eager_scale_error_still_regresses():
    r = torch.linspace(1, 2, 1024, dtype=torch.float64).half().double()
    e = r.half()
    must_fail(
        lambda: assert_no_regression((r * 1.1).half(), e, r, "exact eager"), "10% off"
    )


def check_non_finite_listed(worse_of) -> None:
    r, e = case()
    bad = e.clone()
    bad[3] = float("nan")
    assert "non-finite" in worse_of(compare(bad, e, r))


def test_non_finite_kernel_is_worse():
    check_non_finite_listed(worse)


def test_negative_nan_compares_not_worse():
    def stats_only(table):
        k, e = table["kernel"], table["eager"]
        return [s for s in k if s not in ("finite", "cos") and k[s] > e[s]]

    must_fail(lambda: check_non_finite_listed(stats_only), "NaN stats compare False")


def test_table_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_PRECISION_DIR", str(tmp_path))
    r, e = case()
    assert_no_regression(e, e, r, "written", test="stats_check")
    rows = (tmp_path / "stats_check.jsonl").read_text().splitlines()
    row = json.loads(rows[-1])
    assert row["tensor"] == "written" and "kernel" in row and "eager" in row
    assert {"cos", "abs_mean", "rel_median", "abs_max"} <= row["kernel"].keys()
