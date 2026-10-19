"""The online-cache writer is freed by reference counting alone: a closed CacheWriter
over online latent and text caches is dead right after `del`, with the cyclic collector
off; with a bound job method on it again it stays alive."""

import types
from functools import partial

import pytest

from kohakumoss_data import CacheWriter, DataConfig, MossDataset
from tests._util.encode_fns import ByteCodes
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu


def _cfg(tmp_path) -> DataConfig:
    return DataConfig(
        datasets=[{"source": "synthetic", "count": 4, "seed": 2}],
        resolution_mode="square_crop",
        batching="count",
        batch_size=2,
        target_area=256 * 256,
        latent_cache="online",
        text_cache="online",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=1,
        text_drop=0.0,
        text_encoder_id=ByteCodes.encoder_id,
    )


def _writer(ds: MossDataset, strong: bool = False) -> tuple:
    writer = CacheWriter(ds)
    if strong:
        writer._take_text = types.MethodType(CacheWriter._text_jobs, writer)
    assert writer._take_latents is CacheWriter._latent_jobs
    writer.close()
    return (writer,)


def test_cache_writer_is_freed_on_del(tmp_path):
    ds = MossDataset(_cfg(tmp_path))
    check_released(partial(_writer, ds))
    must_fail(
        lambda: check_released(partial(_writer, ds, strong=True)),
        "a bound job method on the writer",
    )
