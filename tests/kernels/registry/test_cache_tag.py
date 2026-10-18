"""The compile-cache key tag: set on the first kernel build, changed by any change to a
source that registers a custom op, appended to (never replacing) a tag already set."""

import pytest
import torch

from kohakumoss_kernels import cache_tag, registry
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OP = (
    "backends/x/norm.py",
    '@custom_op("kohakumoss::x_bwd")\ndef f(g, x, eps: float): ...',
)


def _differs(a: str, b: str) -> None:
    assert a != b, a


def test_tag_is_set_on_the_first_build():
    registry.load_backends()
    assert cache_tag.tag()
    tags = torch.compiler.config.cache_key_tag.split(cache_tag.SEP)
    assert cache_tag.tag() in tags
    assert any(path.endswith("norm.py") for path, _ in cache_tag.op_sources())


def test_tag_follows_the_registering_sources():
    base = cache_tag.source_tag("0.1.0", [OP])
    changed = (OP[0], OP[1].replace(", eps: float", ""))
    _differs(base, cache_tag.source_tag("0.1.0", [changed]))
    _differs(base, cache_tag.source_tag("0.1.1", [OP]))
    assert base == cache_tag.source_tag("0.1.0", [OP])
    must_fail(lambda: _differs(base, cache_tag.source_tag("0.1.0", [OP])), "same")


def test_an_existing_tag_is_kept():
    assert cache_tag.with_tag("", "k") == "k"
    assert cache_tag.with_tag("user", "k") == "user|k"
    assert cache_tag.with_tag("user|k", "k") == "user|k"
