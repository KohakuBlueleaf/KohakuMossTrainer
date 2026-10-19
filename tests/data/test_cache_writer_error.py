"""A failed background cache write surfaces at the next hand-over to the writer (and at
close, which still stops the thread), not only at the next flush."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss_data.cache.writer import CacheWriter
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OFF = SimpleNamespace(mode="off")


def _failing_writer() -> CacheWriter:
    writer = CacheWriter(SimpleNamespace(latents=OFF, texts=OFF))
    writer._take_latents = lambda w, hb, latents: [("latent", "k", torch.zeros(2))]
    writer(None, None, None, None)
    writer._queue.join()
    assert writer.error is not None, "the constructed write did not fail"
    return writer


def check_surfaces(writer: CacheWriter) -> None:
    try:
        writer(None, None, None, None)
    except RuntimeError as err:
        assert "write-back failed" in str(err), err
        return
    raise AssertionError("a failed write did not surface at the next hand-over")


def test_error_surfaces_at_the_next_write_and_close():
    writer = _failing_writer()
    check_surfaces(writer)
    with pytest.raises(RuntimeError):
        writer.close()
    assert not writer._thread.is_alive()


def test_negative_error_kept_until_flush():
    writer = _failing_writer()
    writer.raise_error = lambda: None
    writer._take_latents = lambda w, hb, latents: []
    must_fail(lambda: check_surfaces(writer), "no check at hand-over")
