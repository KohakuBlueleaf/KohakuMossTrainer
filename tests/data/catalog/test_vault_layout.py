"""The catalog file on KohakuVault 1.x: the planning columns read back as written (the
`prompt_chars` column included), and a file missing fixed columns is refused with the
re-import message."""

import kohakuvault as kv
import pytest

from kohakumoss_data.catalog.vault import FIXED, is_catalog_vault
from kohakumoss_data.format.catalog import REIMPORT, read_catalog, write_catalog
from kohakumoss_data.types import CatalogRow
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _rows() -> list[CatalogRow]:
    rows = []
    for i in range(5):
        row = CatalogRow(
            id=0, locator=f"a/{i}", w=64 + i, h=32, tags=["x"] * i, weight=0.5 + i
        )
        row.extra["prompt_chars"] = 1000 + i
        rows.append(row)
    return rows


def test_columns_read_back(tmp_path):
    path = str(tmp_path / "catalog.db")
    assert write_catalog(path, _rows()) == 5
    cat = read_catalog(path)
    assert cat.w.tolist() == [64, 65, 66, 67, 68]
    assert cat.chars["prompt_chars"].tolist() == [1000, 1001, 1002, 1003, 1004]
    assert cat.chars["tags_chars"].tolist() == [
        len(", ".join(["x"] * i)) for i in range(5)
    ]
    assert cat.weight.tolist() == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5])
    assert cat[3].locator == "a/3" and cat[3].extra["prompt_chars"] == 1003


def test_partial_layout_is_refused(tmp_path):
    path = str(tmp_path / "old.db")
    cols = kv.ColumnVault(path)
    for name in FIXED[:5]:
        cols.ensure(name, "i64")
    cols.close()
    with pytest.raises(ValueError) as err:
        read_catalog(path)
    assert REIMPORT in str(err.value)

    def check(layout_ok):
        assert not layout_ok(path)

    def only_live(p):
        return "live" in {
            c[0] for c in kv.ColumnVault(p, read_only=True).list_columns()
        }

    check(is_catalog_vault)
    must_fail(lambda: check(only_live), "a layout check on the live column alone")
