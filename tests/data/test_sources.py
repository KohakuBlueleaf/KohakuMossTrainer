"""Sources, kohya repeat modes, sidecars, and the moss_shard byte layout."""

import os
import struct

import pytest

from kohakumoss_data.format.catalog import read_catalog
from kohakumoss_data.format.shard import RECORD
from kohakumoss_data.importer import import_source
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.sources.kohya import parse_folder
from tests._util.negative import must_fail
from tests.data import fixtures

pytestmark = pytest.mark.cpu


def _weights(rows) -> dict[str, float]:
    return {r.locator: r.weight for r in rows}


def check_kohya_weights(rows, expected: dict[str, float]) -> None:
    got = {loc.split("/")[0]: w for loc, w in _weights(rows).items()}
    assert got == expected, got


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("first", {"3_2_alpha": 3.0, "1_beta": 1.0, "7_gamma": 7.0}),
        ("product", {"3_2_alpha": 6.0, "1_beta": 1.0, "7_gamma": 7.0}),
        ("ignore", {"3_2_alpha": 1.0, "1_beta": 1.0, "7_gamma": 1.0}),
    ],
)
def test_kohya_repeat_modes(tmp_path, mode, expected):
    layout = fixtures.make_kohya(str(tmp_path))
    src = SOURCES.get("kohya")(
        str(tmp_path), repeat_prefix=mode, artist_from_folder="name"
    )
    rows = list(src.scan())
    assert len(rows) == sum(layout.values())
    check_kohya_weights(rows, expected)
    other = "product" if mode == "first" else "first"
    if expected != {"3_2_alpha": 1.0, "1_beta": 1.0, "7_gamma": 1.0}:
        wrong = SOURCES.get("kohya")(str(tmp_path), repeat_prefix=other)
        must_fail(lambda: check_kohya_weights(list(wrong.scan()), expected), other)


def test_kohya_fields(tmp_path):
    fixtures.make_kohya(str(tmp_path))
    src = SOURCES.get("kohya")(str(tmp_path), artist_from_folder="name")
    rows = {r.locator: r for r in src.scan()}
    assert "7_gamma/orphan.txt" not in rows
    row = rows["3_2_alpha/img1.png"]
    assert (row.w, row.h) == (80, 48)
    assert row.artist == "alpha"
    assert row.tags == ["tag_a", "tag\\(b\\)", "3_2_alpha_1"]
    assert src.open(row.locator) == fixtures.png_bytes(80, 48, 1)
    regex = SOURCES.get("kohya")(str(tmp_path), artist_from_folder=r"^(\w)a")
    assert {r.artist for r in regex.scan()} == {None, "g"}
    assert parse_folder("plain", "first") == (1.0, "plain")


def test_folder_sidecars(tmp_path):
    n = fixtures.make_folder(str(tmp_path))
    rows = {r.locator: r for r in SOURCES.get("folder")(str(tmp_path)).scan()}
    assert len(rows) == n
    assert rows["a/x.png"].caption_nl == "a red square"
    y = rows["b/y.png"]
    assert y.has_alpha and y.tags == ["t1", "t2"] and y.artist == "someone"
    assert y.json == {"layout": {"k": 1}}


def test_jsonl(tmp_path):
    path = fixtures.make_jsonl(str(tmp_path))
    rows = list(SOURCES.get("jsonl")(path).scan())
    assert [(r.w, r.h, r.weight) for r in rows] == [(40, 24, 2.0), (24, 40, 1.0)]
    assert rows[0].tags == ["x", "y"] and rows[1].caption_nl == "tall image"
    assert rows[1].rating == "g"


def test_webdataset(tmp_path):
    images = fixtures.make_tar(str(tmp_path / "shards" / "a.tar"))
    src = SOURCES.get("webdataset")(str(tmp_path / "shards" / "*.tar"))
    rows = list(src.scan())
    assert [r.tags for r in rows] == [
        ["wd_0", "other"],
        ["wd_1", "other"],
        ["wd_2", "other"],
    ]
    assert [src.open(r.locator) for r in rows] == images


def test_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = [fixtures.png_bytes(16, 32, 5), fixtures.png_bytes(32, 16, 6)]
    table = pa.table({"img": data, "tags": [["a"], ["b", "c"]], "weight": [1.0, 3.0]})
    pq.write_table(table, str(tmp_path / "d.parquet"))
    src = SOURCES.get("parquet")(
        str(tmp_path / "d.parquet"), image_col=None, bytes_col="img"
    )
    rows = list(src.scan())
    assert [(r.w, r.h, r.weight, r.tags) for r in rows] == [
        (16, 32, 1.0, ["a"]),
        (32, 16, 3.0, ["b", "c"]),
    ]
    assert [src.open(r.locator) for r in rows] == data


def check_shard_layout(root: str, originals: dict[str, bytes], rows) -> None:
    """Witness .idx records and .bin bytes against the original files directly."""
    for row in rows:
        shard, index = (int(p) for p in row.locator.split(":"))
        with open(os.path.join(root, "shards", f"{shard:05d}.idx"), "rb") as f:
            f.seek(index * 32)
            raw = f.read(32)
        offset, size, w, h, codec, flags = struct.unpack_from("<QIHHBB", raw, 0)
        with open(os.path.join(root, "shards", f"{shard:05d}.bin"), "rb") as f:
            f.seek(offset)
            data = f.read(size)
        assert data == originals[row.extra["origin"]]
        assert (w, h) == (row.w, row.h) and codec == 2
        assert flags == int(row.has_alpha)


def test_moss_shard_layout(tmp_path):
    src_root = tmp_path / "src"
    fixtures.make_kohya(str(src_root))
    src = SOURCES.get("kohya")(str(src_root))
    originals = {r.locator: src.open(r.locator) for r in src.scan()}
    out = str(tmp_path / "shard")
    assert import_source(src, out, shard_bytes=200) == len(originals)
    rows = read_catalog(os.path.join(out, "catalog.db")).rows
    assert RECORD.size == 32
    assert len({r.locator.split(":")[0] for r in rows}) > 1
    check_shard_layout(out, originals, rows)
    native = SOURCES.get("moss_shard")(out)
    assert [native.open(r.locator) for r in native.scan()] == [
        originals[r.extra["origin"]] for r in rows
    ]
    broken = [
        type(r)(
            **{**r.__dict__, "extra": {**r.extra, "origin": rows[0].extra["origin"]}}
        )
        for r in rows
    ]
    must_fail(lambda: check_shard_layout(out, originals, broken), "swapped origin")
