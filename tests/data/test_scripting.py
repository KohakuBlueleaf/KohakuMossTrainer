"""Factory-path resolution used by the cache scripts."""

import pytest

from kohakumoss_data.cache.random_encoders import RandomTextEncoder
from kohakumoss_data.scripting import build_encoder, import_object
from kohakumoss_data.sources.kohya import KohyaSource

pytestmark = pytest.mark.cpu


def test_import_forms():
    assert import_object("kohakumoss_data.sources.kohya.KohyaSource") is KohyaSource
    assert import_object("kohakumoss_data.sources.kohya:KohyaSource") is KohyaSource
    method = import_object("kohakumoss_data.sources.kohya.KohyaSource.scan")
    assert method is KohyaSource.scan
    assert import_object("kohakumoss_data.sources.kohya:KohyaSource.scan") is method
    with pytest.raises(AttributeError):
        import_object("kohakumoss_data.sources.kohya.KohyaSource.nope")
    with pytest.raises(ImportError):
        import_object("no_such_pkg.thing")


def test_build_encoder():
    spec = "kohakumoss_data.cache.random_encoders.RandomTextEncoder"
    enc = build_encoder(spec, {"dim": 8})
    assert isinstance(enc, RandomTextEncoder) and enc(["a"])[0].shape == (13, 8)
    with pytest.raises(TypeError):
        build_encoder("kohakumoss_data.seeds.seed_of", {})
