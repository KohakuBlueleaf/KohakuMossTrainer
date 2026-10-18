"""Kind.stacking as one enum, model-free: every registered kind carries a Stacking
member, the STACKING tuple is its values, and `register` refuses an unknown mode."""

from dataclasses import replace

import pytest

from kohakumoss.adapters.kinds import REGISTRY
from kohakumoss.adapters.kinds.record import STACKING, Stacking, register
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def assert_stacking(kinds, values) -> None:
    assert values == tuple(s.value for s in Stacking)
    assert kinds
    for kind in kinds:
        assert isinstance(kind.stacking, Stacking), kind.name
        assert kind.stacking.value in values


def test_every_kind_uses_the_enum():
    assert_stacking(list(REGISTRY.values()), STACKING)


def test_register_refuses_unknown_stacking():
    kind = next(iter(REGISTRY.values()))
    with pytest.raises(ValueError):
        register(replace(kind, stacking="sideways"))


def test_stacking_check_fails_on_broken_records():
    kind = next(iter(REGISTRY.values()))
    plain = replace(kind, stacking="additive")
    must_fail(lambda: assert_stacking([plain], STACKING), "plain string stacking")
    must_fail(lambda: assert_stacking([kind], STACKING[:-1]), "value dropped")
