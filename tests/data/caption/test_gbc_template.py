"""`gbc` caption template, model free: the metadata rules at every boundary (the danbooru
example's own rules), box mapping into a crop, format draws and their renormalisation,
dropout rates, the budget fit keeping the trailing metadata, and whole captions of a
small on-disk GBC dataset."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kohakumoss_data.caption.gbc import (
    Format,
    GbcTemplate,
    Parts,
    Rate,
    crop_box,
    drop_each,
    rules_of,
)
from kohakumoss_data.sources.gbc import GbcSource
from kohakumoss_data.types import CatalogRow, RowGeometry
from tests._util.gbc import ROOT_LONG, ROOT_SHORT, write_dataset
from tests._util.negative import must_fail

pytest.importorskip("kohakugbc_data")
pytestmark = pytest.mark.cpu

EXAMPLE = (
    Path(__file__).parents[3] / "examples" / "gbc-danbooru" / "caption_danbooru.py"
)


def test_mix_example_shares():
    mix = runpy.run_path(str(EXAMPLE.with_name("caption_danbooru_mix.py")))
    t = GbcTemplate(**_kwargs(mix["CAPTION"]))
    kept = 1 - mix["TEXT_DROP"]
    total = sum(f.weight for f in t.formats)
    share = {f.sections: kept * f.weight / total for f in t.formats}
    heavy = next(f for f in t.formats if f.requires == ("artist",))
    assert heavy.sections == ("artist", "quality", "era", "rating")
    assert share[heavy.sections] == pytest.approx(0.4)
    graph = [s for s in share if "graph" in s]
    assert sum(share[s] for s in graph) == pytest.approx(0.2)
    assert sum(share.values()) + mix["TEXT_DROP"] == pytest.approx(1.0)
    assert heavy.drop["artist"] == Rate(0.0, 0.0)


def example_caption() -> dict:
    return runpy.run_path(str(EXAMPLE))["CAPTION"]


def _kwargs(caption: dict) -> dict:
    return {k: v for k, v in caption.items() if k != "template"}


def check_rules(rules: dict) -> None:
    def q(pct):
        return rules["quality"].items({"selection": {"quality_pct": pct}})

    assert q(0.97) == ["masterpiece"] and q(0.9699) == ["best quality"]
    assert q(0.90) == ["best quality"] and q(0.75) == ["great quality"]
    assert q(0.7499) == [] and q(0.45) == [] and q(0.4499) == ["low quality"]
    assert q(0.10) == ["bad quality"] and q(0.0999) == ["worst quality"]
    assert rules["quality"].items({}) == [] and q("n/a") == []

    def era(year):
        return rules["era"].items({"year": year})

    assert era(2026) == era(2025) == ["newest"] and era(2024) == ["recent"]
    assert era(2020) == ["mid"] and era(2019) == ["early"] and era(2015) == ["early"]
    assert era(2014) == ["old"] and era(2005) == ["old"] and era(2004) == []

    def rating(r):
        return rules["rating"].items({"rating": r})

    assert rating("general") == ["safe"] and rating("sensitive") == ["sensitive"]
    assert rating("questionable") == ["nsfw"]
    assert rating("explicit") == ["nsfw", "explicit"]
    assert rules["artist"].items({"artist": ["a", "b"]}) == ["a", "b"]
    assert rules["artist"].drop == Rate(0.2, 0.2) and rules["quality"].drop == Rate(
        0, 0
    )


def test_example_rules_at_every_boundary():
    rules = rules_of(example_caption()["meta"])
    check_rules(rules)
    lowest_first = replace(
        rules["quality"], bins=tuple(reversed(rules["quality"].bins))
    )
    must_fail(lambda: check_rules({**rules, "quality": lowest_first}), "bins reversed")
    off_by_one = replace(rules["era"], ranges=((2016, 2019, "early"),))
    must_fail(lambda: check_rules({**rules, "era": off_by_one}), "era range shifted")


def check_crop(mapped) -> None:
    geo = RowGeometry(200, 100, (100, 100), (50, 0, 150, 100))
    inside = mapped((0.3, 0.2, 0.6, 0.8), geo)
    assert inside == pytest.approx((0.1, 0.2, 0.7, 0.8))
    assert mapped((0.0, 0.0, 0.4, 1.0), geo) == pytest.approx((0.0, 0.0, 0.3, 1.0))
    assert mapped((0.0, 0.0, 0.2, 1.0), geo) is None
    assert mapped(None, geo) is None


def test_boxes_follow_the_crop():
    check_crop(crop_box)
    must_fail(lambda: check_crop(lambda box, geo: box), "boxes not mapped")


def _parts(nl: bool = True, tags: bool = True) -> Parts:
    return Parts(
        meta={"quality": ["masterpiece"], "artist": ["a"]},
        tags={"meta:general": ["x", "y"] if tags else []},
        nl_long="long caption" if nl else None,
        nl_short="short",
        dag=None,
        short={},
        canvas=(64, 64),
    )


def _template(**kw) -> GbcTemplate:
    caption = {
        "formats": [
            {"weight": 2, "sections": ["artist", "tags", "quality"]},
            {"weight": 2, "sections": ["artist", "nl", "quality"]},
            {"weight": 3, "sections": ["artist", "tags", "nl", "quality"]},
        ],
        "meta": {"artist": {"field": "artist"}, "quality": {"field": "q"}},
        "tags": {"sources": [{"from": "meta:general"}], "drop": 0.0},
        **kw,
    }
    return GbcTemplate(**caption)


def check_draws(choose, parts, want: dict) -> None:
    rng = np.random.default_rng(0)
    n = 6000
    counts: dict = {}
    for _ in range(n):
        f = choose(parts, rng)
        counts[f.sections] = counts.get(f.sections, 0) + 1
    assert set(counts) == set(want), counts
    for sections, p in want.items():
        sd = (p * (1 - p) / n) ** 0.5
        assert abs(counts[sections] / n - p) <= 5 * sd, (sections, counts)


def test_formats_draw_by_weight_and_skip_missing_sections():
    t = _template()
    a, b, c = (f.sections for f in t.formats)
    check_draws(t.choose, _parts(), {a: 2 / 7, b: 2 / 7, c: 3 / 7})
    check_draws(t.choose, _parts(nl=False), {a: 1.0})
    check_draws(t.choose, _parts(tags=False), {b: 1.0})

    def ignores_missing(parts, rng):
        return t.formats[int(rng.integers(len(t.formats)))]

    must_fail(
        lambda: check_draws(ignores_missing, _parts(nl=False), {a: 1.0}),
        "a format with a missing section drawn",
    )


def check_rate(drop, p: float) -> None:
    rng = np.random.default_rng(1)
    items = [str(i) for i in range(20000)]
    kept = len(drop(items, p, rng)) / len(items)
    sd = (p * (1 - p) / len(items)) ** 0.5
    assert abs((1 - kept) - p) < 5 * sd, kept


def test_tags_drop_independently_at_their_rate():
    check_rate(drop_each, 0.2)
    must_fail(lambda: check_rate(lambda items, p, rng: list(items), 0.2), "no drop")


def test_section_drop_and_trailing_metadata():
    t = _template(
        meta={"artist": {"field": "artist", "drop": 0.2}, "quality": {"field": "q"}}
    )
    fmt = Format(1.0, ("artist", "tags", "quality"))
    rng = np.random.default_rng(2)
    texts = []
    for _ in range(4000):
        meta = t.rates(fmt, rng).meta
        texts.append(t.draw(fmt, _parts(), "meta:general", rng, (0.0, 0.0, 0.5), meta))
    dropped = sum(not s.startswith("a,\n") for s in texts) / len(texts)
    assert abs(dropped - 0.2) < 5 * (0.16 / 4000) ** 0.5
    assert all(s.endswith(",\nmasterpiece") for s in texts)


def check_range(rate: Rate, lo: float, hi: float) -> None:
    rng = np.random.default_rng(4)
    draws = np.array([rate.draw(rng) for _ in range(4000)])
    assert draws.min() >= lo and draws.max() <= hi
    assert draws.min() < lo + 0.02 and draws.max() > hi - 0.02
    assert abs(draws.mean() - (lo + hi) / 2) < 0.02


def test_rates_fixed_or_drawn_per_caption():
    check_range(Rate.of([0.0, 0.4]), 0.0, 0.4)
    must_fail(lambda: check_range(Rate.of(0.2), 0.0, 0.4), "a fixed rate as a range")
    rng = np.random.default_rng(9)
    state = rng.bit_generator.state
    assert Rate.of(0.3).draw(rng) == 0.3 and rng.bit_generator.state == state
    for bad in (1.5, [0.5, 0.2], [-0.1, 0.3]):
        with pytest.raises(ValueError):
            Rate.of(bad)
    t = _template(tags={"sources": [{"from": "meta:general"}], "drop": [0.0, 0.6]})
    rates = [t.rates(t.formats[0], rng).tag_drop for _ in range(2000)]
    assert min(rates) < 0.01 and max(rates) > 0.59


def test_format_requires_and_drop_override():
    t = _template(
        formats=[
            {
                "weight": 4,
                "sections": ["artist", "quality"],
                "requires": ["artist"],
                "drop": {"artist": 0.0},
            },
            {"weight": 1, "sections": ["artist", "tags", "quality"]},
        ],
        meta={"artist": {"field": "artist", "drop": 0.5}, "quality": {"field": "q"}},
    )
    heavy, normal = (f.sections for f in t.formats)
    check_draws(t.choose, _parts(), {heavy: 0.8, normal: 0.2})
    no_artist = replace(_parts(), meta={"quality": ["masterpiece"]})
    check_draws(t.choose, no_artist, {normal: 1.0})
    rng = np.random.default_rng(3)
    assert all(t.rates(t.formats[0], rng).meta["artist"] == 0.0 for _ in range(50))
    assert all(t.rates(t.formats[1], rng).meta["artist"] == 0.5 for _ in range(50))
    must_fail(
        lambda: check_draws(t.choose, no_artist, {heavy: 0.8, normal: 0.2}),
        "requires ignored",
    )


@pytest.fixture
def gbc_root(tmp_path):
    write_dataset(tmp_path, n=4, per_shard=2)
    return tmp_path


def _bound(root, caption: dict, max_chars: int = 0):
    src = GbcSource(str(root))
    rows = [r for item in src.walk() for r in item.probe()]
    t = GbcTemplate(**_kwargs(caption))
    return t, t.bind(src, None, max_chars), rows


def test_example_captions_of_a_gbc_dataset(gbc_root):
    t, render, rows = _bound(gbc_root, example_caption())
    seen = set()
    for i, row in enumerate(rows):
        for k in range(40):
            text = render(row, np.random.default_rng((i, k)), i)
            assert text == render(row, np.random.default_rng((i, k)), i)
            for raw in ("long_hair", "girl_(series)", "some_artist", "red_coat"):
                assert raw not in text, text
            if "<gbc bbox=yxyx space=abs_centralized canvas=64x32>" in text:
                seen.add("graph")
                assert f"# [image] {ROOT_LONG}" in text
            if ROOT_SHORT in text and "<gbc" not in text:
                seen.add("short")
            tail = ["safe", "sensitive", "nsfw", "nsfw, explicit"][i % 4]
            assert text.endswith(tail), text
            assert "masterpiece" in text if i == 0 else True
    assert seen == {"graph", "short"}


def check_tag_filters(texts: list[str]) -> None:
    for text in texts:
        assert "rating:" not in text, text
        assert text.count("some artist") <= 1, text
        assert ".," not in text and ".\n," not in text, text


def test_tag_filters_and_text_separator(gbc_root):
    caption = example_caption()
    _, render, rows = _bound(gbc_root, caption)
    texts = [render(rows[0], np.random.default_rng(s), 0) for s in range(80)]
    assert any("<gbc" in t for t in texts) and any("1girl" in t for t in texts)
    check_tag_filters(texts)
    loose = {
        **caption,
        "tags": {**caption["tags"], "exclude": [], "dedupe": []},
        "text_sep": ",\n",
    }
    _, unfiltered, _ = _bound(gbc_root, loose)
    raw = [unfiltered(rows[0], np.random.default_rng(s), 0) for s in range(80)]
    must_fail(lambda: check_tag_filters(raw), "no exclude / dedupe, comma after text")


def test_fit_redraws_until_it_fits_and_keeps_the_metadata(gbc_root):
    caption = example_caption()
    _, free, rows = _bound(gbc_root, caption)
    t, fit, _ = _bound(gbc_root, caption, max_chars=10_000)
    graph_only = {
        **caption,
        "formats": [f for f in caption["formats"] if "graph" in f["sections"]],
    }
    _, tight, _ = _bound(gbc_root, graph_only, max_chars=int(260 / 0.75))
    row = rows[0]
    text = free(row, np.random.default_rng(5), 0)
    assert fit(row, np.random.default_rng(5), 0) == text
    small = [tight(row, np.random.default_rng(s), 0) for s in range(30)]

    def check(texts):
        assert all(len(s) <= 260 or "<gbc" not in s for s in texts)
        assert all(s.endswith("masterpiece,\nold,\nsafe") for s in texts), texts[0]

    check(small)
    _, graph_free, _ = _bound(gbc_root, graph_only)
    longest = graph_free(row, np.random.default_rng(0), 0)
    assert len(longest) > 300 and "<gbc" in longest
    must_fail(lambda: check([longest[:259]]), "a fit that truncates")


def test_source_rows_and_records(gbc_root):
    src = GbcSource(str(gbc_root))
    items = src.walk()
    assert len(items) == 2
    rows = [r for item in items for r in item.probe()]
    assert [r.locator for r in rows] == [
        "shard_00000/100",
        "shard_00000/101",
        "shard_00001/102",
        "shard_00001/103",
    ]
    assert all((r.w, r.h) == (64, 32) and r.extra["prompt_chars"] > 100 for r in rows)
    rec = src.record("shard_00001/103")
    assert rec.meta["rating"] == "explicit" and rec.graph.num_vertices == 4
    assert src.open("shard_00000/101")[:4] == b"RIFF"
    assert isinstance(rows[0], CatalogRow)
