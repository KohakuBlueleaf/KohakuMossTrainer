"""The catalog table: docs/concepts/profiles.md and `kohakumoss-config list` show the
generated table, every entry has an estimate, short names are catalog entries, and no
VRAM layer sets precision."""

from pathlib import Path

import pytest

from kohakumoss.cli.config import main as config_main
from kohakumoss.cli.config.validate import load_knobs
from kohakumoss.memory import catalog_table, profiles
from kohakumoss.memory.catalog_table import markdown, rows
from kohakumoss.memory.knob_map import PRECISION_KNOBS
from kohakumoss.memory.profiles import CATALOG, ORDER, PROFILES, catalog_name
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "concepts" / "profiles.md"
BEGIN, END = "<!-- catalog:begin -->", "<!-- catalog:end -->"
SHORT = {
    "speed": "nockpt-resident",
    "balanced": "ckpt-ring16",
    "low": "ckpt-ring4-actoff-optstream",
    "minimal": "ckpt-ring2-actoff-optstream-gradoff",
}


def _doc_table(text: str) -> str:
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def test_docs_table_is_the_generated_table():
    def check(text: str) -> None:
        assert _doc_table(text) == markdown()

    text = DOC.read_text(encoding="utf-8")
    check(text)
    lines = text.splitlines()
    dropped = "\n".join(line for line in lines if "`ckpt-ring8-optcpu`" not in line)
    must_fail(lambda: check(dropped), "docs table missing one catalog row")


def test_every_entry_has_an_estimate(monkeypatch):
    def check(table: list[tuple[str, ...]]) -> None:
        assert len(table) == len(ORDER) + len(CATALOG)
        assert not [r for r in table if any(c.startswith("n/a") for c in r[-2:])]

    check(rows())
    monkeypatch.setitem(PROFILES, "bad", dict(stream="ring", stream_resident=1))
    lora, _ = catalog_table.workloads()
    broken = [(*r[:-2], catalog_table._peak(lora, "bad"), r[-1]) for r in rows()]
    must_fail(lambda: check(broken), "an entry that does not resolve")


def test_short_names_are_catalog_entries(monkeypatch):
    def check() -> None:
        assert {n: catalog_name(n) for n in ORDER} == SHORT

    check()
    off_catalog = dict(checkpoint="block", stream="ring", stream_resident=3)
    monkeypatch.setitem(profiles.PROFILES, "balanced", off_catalog)
    must_fail(check, "balanced off the catalog")


def _precision_set(k: dict) -> list[str]:
    named = [n for n in (*PRECISION_KNOBS, "VAE_DTYPE") if k.get(n) is not None]
    fields = set(PRECISION_KNOBS.values()) & set(k.get("MEMORY_OVERRIDES") or {})
    text = (
        ["TEXT_ENCODE.precision"] if "precision" in (k.get("TEXT_ENCODE") or {}) else []
    )
    return named + sorted(fields) + text


@pytest.mark.parametrize("layer", sorted((ROOT / "configs/moss/vram").glob("*.py")))
def test_vram_layers_set_no_precision(layer):
    def check(k: dict) -> None:
        assert _precision_set(k) == [], layer.name

    k = load_knobs(layer)
    check(k)
    quant = {**k, "MEMORY_OVERRIDES": {"base_precision": "int8"}}
    must_fail(lambda: check(quant), "a VRAM layer with an int8 base")


def test_config_list_prints_the_table(capsys, monkeypatch):
    def check() -> None:
        config_main.list_presets(ROOT)
        short = capsys.readouterr().out
        assert markdown(names=ORDER) in short and "list --all" in short
        assert f"`{next(iter(CATALOG))}`" not in short
        config_main.list_presets(ROOT, everything=True)
        assert markdown() in capsys.readouterr().out

    check()
    monkeypatch.setattr(config_main, "profile_table", lambda names=None: "")
    must_fail(check, "list without the profile table")
