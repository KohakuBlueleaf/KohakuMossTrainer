"""The release stage: its rows per pool (cached, one no-cache row, fp8 on sm89+, the
vram_8g layer under its cap in every cache mode, vram_6g cached), every knob a
training-script knob, each row's cache tag its LATENT_CACHE / TEXT_CACHE pair, and one
cache build per encoder dtype a host's cards resolve to, pinned to that dtype."""

from pathlib import Path

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.plan.config import Card, Host, Options
from kohakumoss_bench.plan.knobs import (
    TRAIN,
    cache_knobs,
    encoder_dtype,
    script_defaults,
)
from kohakumoss_bench.plan.matrix import CACHE_MODES
from kohakumoss_bench.run import release
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
REPO = Path(__file__).resolve().parents[3]
CARDS = {
    "5090": (Card(0, "5090", 32.0, 120), 10),
    "3090": (Card(0, "3090", 24.0, 86), 9),
    "2080ti": (Card(2, "2080ti", 22.0, 75), 9),
    "v100": (Card(3, "v100", 16.0, 70), 9),
}


def host() -> Host:
    return Host(
        name="h",
        root="/r",
        python="python",
        weights={"source": "/w/base", "stage": "/s/base"},
        datasets=[{"source": "kohya", "path": "/d"}],
        cache_dir="/c",
    )


def check_release(rows: list, card: Card, count: int) -> None:
    assert len(rows) == count, (card.name, [r.name for r in rows])
    assert rows[0].timeout_s == Options().timeout_first_s
    known = set(script_defaults(REPO / TRAIN))
    for r in rows:
        assert r.kind == "release" and r.script == TRAIN
        assert not set(r.knobs) - known, (r.name, sorted(set(r.knobs) - known))
    for r in rows:
        caches = CACHE_MODES[r.tags["cache"]]
        assert {k: r.knobs[k] for k in caches} == caches, (r.name, r.tags["cache"])
    plain = [r for r in rows if not r.cap_gib and r.tags["cache"] == "nocache"]
    assert [r.name.split("-", 2)[2] for r in plain] == ["lora32-balanced-nocache"]
    vram = {(r.cap_gib, r.tags["cache"]) for r in rows if r.cap_gib}
    want = {(8.0, mode) for mode in CACHE_MODES} | {(6.0, "latenttext")}
    assert vram == want and len(vram) == len([r for r in rows if r.cap_gib]), vram
    assert any(r.tags["base"] == "fp8_compute" for r in rows) == card.fp8
    if not card.bf16:
        assert {r.tags["precision"] for r in rows} == {"fp16"}


def test_release_rows_per_pool():
    for card, count in CARDS.values():
        check_release(release.queue(host(), Options(), card, (), REPO), card, count)


def test_negative_release_without_the_vram_layers():
    card, count = CARDS["5090"]
    rows = [
        r for r in release.queue(host(), Options(), card, (), REPO) if not r.cap_gib
    ]
    must_fail(lambda: check_release(rows, card, count), "vram rows missing")


def test_negative_release_with_a_mislabelled_cache_mode():
    card, count = CARDS["3090"]
    rows = release.queue(host(), Options(), card, (), REPO)
    latent = next(r for r in rows if r.tags["cache"] == "latent")
    latent.tags["cache"] = "nocache"
    must_fail(lambda: check_release(rows, card, count), "latent row tagged nocache")


def test_negative_release_without_the_vram_cache_modes():
    card, count = CARDS["3090"]
    rows = [
        r
        for r in release.queue(host(), Options(), card, (), REPO)
        if not (r.cap_gib and r.tags["cache"] != "latenttext")
    ]
    must_fail(lambda: check_release(rows, card, count - 2), "vram_8g cache modes")


RUNS = ["release-*-lora32-balanced", "release-*-lora32-vram6g"]


def check_runs_kept(rows: list) -> None:
    names = [r.name for r in rows]
    assert len(names) == 2, names
    assert sorted(n.split("-")[-1] for n in names) == ["balanced", "vram6g"], names
    assert rows[0].timeout_s == Options().timeout_first_s


def test_release_queue_keeps_the_host_runs():
    card, _ = CARDS["3090"]
    picky = Host(**{**vars(host()), "runs": RUNS})
    check_runs_kept(release.queue(picky, Options(), card, (), REPO))
    all_rows = release.queue(host(), Options(), card, (), REPO)
    must_fail(lambda: check_runs_kept(all_rows), "runs patterns ignored")


def check_cache_specs(knobs_of) -> None:
    cards = [Card(0, "3090", 24.0, 86), Card(2, "2080ti", 22.0, 75)]
    builds = {encoder_dtype(c): c for c in cards}
    assert set(builds) == {"bf16", "fp16"}, "one build per encoder dtype"
    for dtype, card in builds.items():
        _, latents = knobs_of(card, "latents")
        _, text = knobs_of(card, "text")
        assert latents["VAE_DTYPE"] == dtype, latents["VAE_DTYPE"]
        assert text["TEXT_ENCODER_KWARGS"]["plan"]["dtype"] == dtype


def test_cache_builds_pin_the_encoder_dtype():
    check_cache_specs(lambda card, kind: cache_knobs(host(), Options(), card, kind))


def test_negative_cache_build_on_auto_dtypes():
    def auto(card, kind):
        script, knobs = cache_knobs(host(), Options(), card, kind)
        if kind == "latents":
            return script, {**knobs, "VAE_DTYPE": "auto"}
        kwargs = dict(knobs["TEXT_ENCODER_KWARGS"], plan={"memory": "auto"})
        return script, {**knobs, "TEXT_ENCODER_KWARGS": kwargs}

    with pytest.raises((AssertionError, KeyError)):
        check_cache_specs(auto)
