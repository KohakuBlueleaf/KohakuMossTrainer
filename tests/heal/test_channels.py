"""Channel stats, the shrink scales and table, reader-norm sites and their taps."""

import numpy as np
import pytest
import torch

from kohakumoss.heal.channels import (
    peak_ratio,
    ratio_summary,
    shrink_scales,
    shrink_table,
    stream_stats,
    top_channels,
    writer_names,
)
from kohakumoss.heal.sites import NormTaps, norm_path, site_names
from kohakumoss.heal.wnorm import WeightedNorms, anneal_lam
from kohakumoss_kernels.backends.eager.norm import rms_norm
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def test_peak_ratio_and_scales():
    ms = np.ones(16)
    ms[5] = 400.0**2
    other = np.ones(16)
    other[2] = 30.0**2
    peak = peak_ratio({"a": ms, "b": other})
    assert peak[5] == pytest.approx(400.0)
    assert peak[2] == pytest.approx(30.0)
    d = shrink_scales(peak, 16.0)
    assert d[5].item() == pytest.approx(16 / 400)
    assert d[2].item() == pytest.approx(16 / 30)
    assert d[0].item() == 1.0
    assert top_channels(peak, 2) == [[5, 400.0], [2, 30.0]]


def test_sites():
    names = site_names(2, 42)
    assert len(names) == 89
    assert names[:2] == ["t0.1", "t0.2"] and names[4] == "j0.1" and names[-1] == "h"
    assert norm_path("t1.2") == "text_embedder.blocks.1.norm2"
    assert norm_path("j10.1") == "layers.10.norm1"
    assert norm_path("h") == "final_norm_video"


def _table_check(table_of):
    """A writer row times D through the weighted norm (lam 1/D^2) with the table's gain
    equals the original stream through the plain norm with the original gain."""
    dim = 8
    d = torch.ones(dim)
    d[1] = 0.125
    table = table_of(d)
    x = torch.randn(5, dim, dtype=torch.float64)
    w = torch.rand(dim, dtype=torch.float64) + 0.5
    wn = WeightedNorms(dim, "cpu")
    wn.lam = anneal_lam(d, 0.0)
    xs = x * table["layers.0.proj.o.weight"].double()
    got = wn.rms_norm(xs, w * table["layers.0.norm1.weight"].double(), 1e-6)
    torch.testing.assert_close(got, rms_norm(x, w, 1e-6))


def test_shrink_table():
    sites = site_names(2, 3)
    table = shrink_table(2, 3, sites, torch.full((8,), 0.5), {s: 2.0 for s in sites})
    assert set(writer_names(2, 3)) <= set(table)
    assert len(table) == len(writer_names(2, 3)) + len(sites)
    torch.testing.assert_close(table["final_norm_video.weight"], torch.full((8,), 1.0))
    _table_check(lambda d: shrink_table(2, 3, sites, d))


def test_shrink_table_negative():
    sites = site_names(2, 3)
    must_fail(
        lambda: _table_check(
            lambda d: shrink_table(2, 3, sites, d, {s: 1.3 for s in sites})
        ),
        "gains carry r",
    )


def test_stream_stats_and_ratios():
    x = torch.ones(4, 16)
    x[:, 0] = 10.0
    rms, absmax, top8, shrunk = stream_stats(x, torch.tensor([0]))
    assert absmax == 10.0 and rms == pytest.approx(float(np.sqrt((100 + 15) / 16)))
    assert top8 == pytest.approx((100 + 7) / 115)
    assert shrunk == pytest.approx(100 / 115)
    assert stream_stats(x)[3] == 0.0
    s = ratio_summary(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0]))
    assert s["rbar"] == 2.0 and len(s["img"]) == 3 and s["txt"] == [1.0, 1.0, 1.0]


def test_norm_taps_order_and_count():
    seen = []
    taps = NormTaps(["a", "b", "c"], lambda x, w, e: x, lambda x, w, sh, sc, seg, e: x)
    taps.fn = lambda site, x: seen.append(site)
    taps.begin()
    taps.rms_norm(torch.ones(1), None, 0.0)
    taps.rms_norm_mod(torch.ones(1), None, None, None, None, 0.0)
    with pytest.raises(RuntimeError):
        taps.end()
    taps.rms_norm(torch.ones(1), None, 0.0)
    taps.end()
    assert seen == ["a", "b", "c"]
    with pytest.raises(RuntimeError):
        taps.rms_norm(torch.ones(1), None, 0.0)
    taps.fn = None
    taps.begin()
    taps.end()
