"""Caption templates: keep order, dropout, groups, unescape, nl/mix/json."""

import numpy as np
import pytest

from kohakumoss_data.caption import build_template, split_tags, unescape
from kohakumoss_data.types import CatalogRow
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def row(**kw) -> CatalogRow:
    base = dict(id=0, locator="x", w=8, h=8, artist="painter_x")
    base.update(kw)
    return CatalogRow(**base)


def rng(seed=0):
    return np.random.default_rng(seed)


def check_keep_first(text: str, keep: str) -> None:
    assert text.split(", ")[0] == keep, text


def test_tags_keep_shuffle_unescape():
    t = build_template({"template": "tags", "keep": ["artist"], "shuffle": True})
    r = row(tags=["long_hair", "smile", "fate \\(series\\)", "a", "b", "c"])
    outs = {t.render(r, rng(s)) for s in range(8)}
    assert len(outs) > 1
    for text in outs:
        check_keep_first(text, "painter x")
        assert "fate (series)" in text and "long hair" in text
    raw = build_template(
        {"template": "tags", "keep": [], "underscores": False, "unescape": False}
    )
    text = raw.render(r, rng())
    must_fail(lambda: check_keep_first(text, "painter x"), "no keep field")
    assert "fate \\(series\\)" in raw.render(r, rng())


def test_dropout_and_groups():
    r = row(tags=["k1", "|||", "a", "b", "%%", "c", "d"])
    all_drop = build_template({"template": "tags", "keep": [], "tag_dropout": 1.0})
    assert all_drop.render(r, rng()) == "k1"
    groups = build_template(
        {"template": "tags", "keep": [], "group_dropout": 0.5, "shuffle": False}
    )
    seen = {groups.render(r, rng(s)) for s in range(32)}
    assert {"k1", "k1, a, b", "k1, c, d", "k1, a, b, c, d"} <= seen | {"k1, c, d, a, b"}
    assert all(s.startswith("k1") for s in seen)


def test_nl_mix_json():
    r = row(
        tags=["t"], caption_nl="a cat on a mat", json={"layout": [1, 2]}, rating="g"
    )
    nl = build_template({"template": "nl", "keep": ["artist"]})
    assert nl.render(r, rng()) == "painter_x. a cat on a mat"
    fallback = nl.render(row(tags=["t"]), rng())
    assert fallback == "painter x, t"
    mix = build_template({"template": "mix", "p_nl": 0.5})
    assert len({mix.render(r, rng(s)) for s in range(16)}) == 2
    js = build_template({"template": "json", "fields": ["artist", "rating"]})
    assert js.render(r, rng()) == "artist: painter_x\nlayout: 1, 2\nrating: g"
    drop = build_template(
        {
            "template": "json",
            "fields": ["artist"],
            "field_dropout": 1.0,
            "keep": ["artist"],
        }
    )
    assert drop.render(r, rng()) == "artist: painter_x"


def test_tag_helpers():
    assert split_tags(" a, b ,, c ") == ["a", "b", "c"]
    assert unescape("x \\(y\\) \\[z\\]") == "x (y) [z]"
