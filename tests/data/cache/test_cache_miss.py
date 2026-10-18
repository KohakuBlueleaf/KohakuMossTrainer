"""The offline-cache miss error: names the first missing sample, the key knobs that
differ from the nearest recorded cache build (or the run's knobs when the cache
recorded none), and the cache pass that rebuilds it with the run's settings."""

import os

import pytest

from kohakumoss_data import MossDataset
from kohakumoss_data.cache.build import build_latents, build_text
from kohakumoss_data.cache.settings import CacheMiss, settings_path
from tests._util.encode_fns import ByteCodes, PatchMeans
from tests._util.negative import must_fail
from tests.data.cache.test_cache import _cfg

pytestmark = pytest.mark.cpu
BARE = (
    "3 latents missing from the offline cache; run scripts/data/cache_latents.py "
    "with the same data config"
)


def _built(tmp_path):
    cfg = _cfg(tmp_path)
    ds = MossDataset(cfg)
    build_text(ds, ByteCodes(dim=32))
    build_latents(ds, PatchMeans(channels=16))
    return cfg


def _miss(cfg) -> str:
    with pytest.raises(CacheMiss) as err:
        MossDataset(cfg).materialize(0)
    return str(err.value)


def _names_area_and_rebuild(msg: str) -> None:
    assert "TARGET_AREA: 147456 vs 65536" in msg, msg
    assert "MAX_TOKENS" not in msg, msg
    assert "LATENT_ENCODER_ID" not in msg.split("Rebuild")[0], msg
    assert "first: row " in msg and "-> target " in msg, msg
    assert "TARGET_AREA = 147456" in msg, msg
    assert "kogine run scripts/data/cache_latents.py" in msg, msg
    assert "--set TARGET_AREA=147456" in msg, msg


def test_latent_miss_names_the_differing_knobs_and_the_rebuild(tmp_path):
    _built(tmp_path)
    run = _cfg(tmp_path, target_area=384 * 384, max_tokens=8192)
    _names_area_and_rebuild(_miss(run))
    must_fail(lambda: _names_area_and_rebuild(BARE), "message without the key knobs")


def _lists_run_knobs(msg: str) -> None:
    assert "records no build settings" in msg, msg
    assert "TARGET_AREA: 147456" in msg and "RESAMPLE: 'lanczos'" in msg, msg
    assert "--set TARGET_AREA=147456" in msg, msg


def test_unrecorded_cache_lists_the_run_knobs(tmp_path):
    cfg = _built(tmp_path)
    os.remove(settings_path(cfg.cache_dir, "latents"))
    msg = _miss(_cfg(tmp_path, target_area=384 * 384))
    _lists_run_knobs(msg)
    must_fail(lambda: _lists_run_knobs(BARE), "message without the run's knobs")


def _names_encoder(msg: str) -> None:
    assert "text embeddings missing" in msg, msg
    assert "TEXT_ENCODER_ID: 'other-te' vs " in msg, msg
    assert "kogine run scripts/data/cache_text.py" in msg, msg
    assert "--set TEXT_ENCODER_ID=other-te" in msg, msg


def test_text_miss_names_the_encoder(tmp_path):
    _built(tmp_path)
    msg = _miss(_cfg(tmp_path, text_encoder_id="other-te"))
    _names_encoder(msg)
    must_fail(lambda: _names_encoder(BARE), "message without the encoder id")
