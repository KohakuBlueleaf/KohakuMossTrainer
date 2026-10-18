"""The weight-ring spellings (model-free): the catalog, `kohakumoss-config list` and the
text-encoder plan use `ring16` .. `ring2` / `ring`; the older `stream16` .. `stream2`
profile parts and text-encoder memory `stream` resolve once, at knob / plan validation,
to the primary names."""

import re

import pytest

from kohakumoss.encoders.text.plan import MEMORY, text_plan
from kohakumoss.memory.catalog_table import markdown
from kohakumoss.memory.profiles import CATALOG, profile_name, resolve
from kohakumoss.training import knobs as train_knobs
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OLD = "ckpt-stream4-actoff-optstream-gradoff"
NEW = "ckpt-ring4-actoff-optstream-gradoff"


def check_primary(name_of) -> None:
    """Old and new spellings name one catalog entry, in its primary spelling."""
    assert name_of(OLD) == NEW
    assert name_of(NEW) == NEW
    assert name_of("nockpt-stream16-actoff") == "nockpt-ring16-actoff"
    for kept in ("balanced", "auto", "manual", "", None):
        assert name_of(kept) == kept


def _replace_everywhere(name):
    return name.replace("stream", "ring") if isinstance(name, str) else name


def test_old_profile_parts_resolve_to_the_ring_names():
    check_primary(profile_name)
    must_fail(lambda: check_primary(lambda n: n), "aliases not resolved")
    must_fail(lambda: check_primary(_replace_everywhere), "optstream renamed too")
    assert resolve(OLD, device="cpu") == resolve(NEW, device="cpu")
    assert resolve(OLD, device="cpu").profile == NEW
    collected = train_knobs.collect({"MEMORY_PROFILE": OLD, "lower": 1})
    assert collected == {"MEMORY_PROFILE": NEW}


def check_emitted(names, table: str) -> None:
    """Every catalog name and the table use the ring spelling only."""
    assert {"ckpt-ring16", "ckpt-ring2-actoff-optstream-gradoff"} <= set(names)
    assert not [n for n in names if re.search(r"stream\d", n)]
    assert "| ring 16 |" in table and not re.search(r"\| stream \d+ \|", table)


def test_catalog_and_table_emit_the_ring_names():
    check_emitted(list(CATALOG), markdown())
    old = [re.sub(r"ring(\d)", r"stream\1", n) for n in CATALOG]
    must_fail(lambda: check_emitted(old, markdown()), "old names emitted")


def check_text_memory(plan_of) -> None:
    """Text-encoder memory "stream" is the ring placement, and "ring" is the listed value."""
    assert plan_of({"memory": "stream"}).memory == "ring"
    assert plan_of({"memory": "ring"}).memory == "ring"
    assert "ring" in MEMORY and "stream" not in MEMORY


def _unresolved(spec):
    plan = text_plan(spec)
    return plan if spec.get("memory") != "stream" else type(plan)(memory="stream")


def test_text_encoder_stream_is_the_ring():
    check_text_memory(text_plan)
    must_fail(lambda: check_text_memory(_unresolved), "text memory alias unresolved")
