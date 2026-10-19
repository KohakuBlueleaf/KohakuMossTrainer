"""DataConfig knob mapping, cache-dir resolution, field types and build-problem listing."""

import os

import pytest

from kohakumoss_data import (
    DataConfig,
    MossDataset,
    config,
    data_problems,
    resolved_cache_dir,
    type_problems,
)
from kohakumoss_data.scripting import data_config
from tests._util.negative import must_fail
from tests.data import fixtures

pytestmark = pytest.mark.cpu


def test_resolved_cache_dir(tmp_path):
    folder = tmp_path / "ds"
    folder.mkdir()
    manifest = tmp_path / "m" / "meta.jsonl"
    on = dict(latent_cache="offline", text_cache="off")
    assert resolved_cache_dir(DataConfig(cache_dir="x", **on)) == "x"
    assert resolved_cache_dir(
        DataConfig(datasets=[{"path": str(folder)}], **on)
    ) == str(folder / "cache")
    assert resolved_cache_dir(
        DataConfig(datasets=[{"path": str(manifest)}], **on)
    ) == os.path.join(str(tmp_path / "m"), "cache")
    shards = str(tmp_path / "w" / "*.tar")
    assert resolved_cache_dir(DataConfig(datasets=[{"path": shards}], **on)) == (
        os.path.join(str(tmp_path / "w"), "cache")
    )
    off = DataConfig(
        datasets=[{"path": str(folder)}], latent_cache="off", text_cache="off"
    )
    assert resolved_cache_dir(off) == ""
    assert (
        resolved_cache_dir(DataConfig(datasets=[{"source": "synthetic"}], **on)) == ""
    )


def test_from_knobs_resolves_and_dataset_agrees(tmp_path):
    knobs = {
        "DATASETS": [{"source": "folder", "path": str(tmp_path)}],
        "LATENT_CACHE": "online",
        "TEXT_CACHE": "online",
        "CACHE_DIR": "",
        "UNRELATED": 1,
    }
    cfg = DataConfig.from_knobs(knobs)
    assert cfg.cache_dir == str(tmp_path / "cache")
    (tmp_path / "a.png").write_bytes(fixtures.png_bytes(32, 32, 0))
    raw = DataConfig(datasets=[{"source": "folder", "path": str(tmp_path)}])
    raw.latent_cache = raw.text_cache = "online"
    ds = MossDataset(raw)
    assert os.path.dirname(ds.latents.store.path) == str(tmp_path / "cache")


def check_clean(cfg) -> None:
    assert data_problems(cfg) == [], data_problems(cfg)


def test_data_problems():
    good = DataConfig(
        datasets=[{"source": "synthetic"}], latent_cache="off", text_cache="off"
    )
    check_clean(good)
    for field, value in (
        ("resolution_mode", "pack"),
        ("batching", "greedy"),
        ("overlong", "cut"),
        ("cache_dtype", "fp4"),
        ("crop", "left"),
        ("background", "pink"),
        ("datasets", []),
        ("text_cache", "online"),
        ("token_budget", "exact"),
        ("size_step", 24),
    ):
        bad = DataConfig(**{**good.__dict__, field: value})
        must_fail(lambda b=bad: check_clean(b), f"{field}={value}")
    shifted = DataConfig(
        **{**good.__dict__, "latent_cache": "offline", "cache_dir": "c"}
    )
    shifted.resolution_mode = "shifted_square"
    assert any("centre crop" in p for p in data_problems(shifted))


GOOD = dict(datasets=[{"source": "synthetic"}], latent_cache="off", text_cache="off")
TYPED = (
    ("text_drop", 0),
    ("data_wait", 600),
    ("aspect_range", [0.5, 2]),
    ("background", (0, 0, 0)),
    ("background", "black"),
    ("data_rescan", "never"),
    ("data_rescan", True),
)
MISTYPED = (
    ("target_area", "262144"),
    ("max_tokens", 8.0),
    ("seed", True),
    ("flip", 1),
    ("aspect_range", (1.0,)),
    ("aspect_range", (1.0, "2")),
    ("background", 5),
    ("data_rescan", 3),
    ("datasets", [["x"]]),
    ("caption", "tags"),
)


def check_typed(cfg) -> None:
    assert type_problems(cfg) == [], type_problems(cfg)


def check_reported(field: str, cfg) -> None:
    found = type_problems(cfg)
    assert len(found) == 1 and found[0].startswith(f"{field.upper()}="), found
    assert data_problems(cfg) == found, data_problems(cfg)


def test_type_problems(monkeypatch):
    for field, value in TYPED:
        check_typed(DataConfig(**{**GOOD, field: value}))
    for field, value in MISTYPED:
        bad = DataConfig(**{**GOOD, field: value})
        check_reported(field, bad)
        must_fail(lambda b=bad: check_typed(b), f"{field}={value!r}")
    area = DataConfig(**{**GOOD, "target_area": "262144"})
    assert type_problems(area) == ["TARGET_AREA='262144' is not int"]
    monkeypatch.setattr(config, "mismatches", lambda cfg: [])
    must_fail(lambda: check_reported("target_area", area), "no type check")


def test_data_config_exits_on_a_type_problem():
    knobs = {"DATASETS": [{"source": "synthetic"}], "LATENT_CACHE": "off"}
    knobs["TEXT_CACHE"] = "off"
    assert data_config({**knobs, "TARGET_AREA": 262144}).target_area == 262144
    with pytest.raises(SystemExit, match="TARGET_AREA='262144' is not int"):
        data_config({**knobs, "TARGET_AREA": "262144"})
