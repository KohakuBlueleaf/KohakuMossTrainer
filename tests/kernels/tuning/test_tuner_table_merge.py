"""Two processes tuning the same card model share one table file: each write keeps the
rows the other wrote since it loaded the table."""

import pytest

from kohakumoss_kernels.hardware.device import read_json, write_json
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_both_kept(path, table_cls) -> None:
    a, b = table_cls(path), table_cls(path)
    a.put("k1", {"x": 1}, {"x=1": 1.0}, {})
    b.put("k2", {"x": 2}, {"x=2": 2.0}, {})
    a.put_configs({"k3": {"x": 3}})
    rows = read_json(path)
    assert set(rows) == {"k1", "k2", "k3"}, sorted(rows)
    assert rows["k2"]["config"] == {"x": 2}


class OverwritingTable(tuner.TuneTable):
    """The table writing only its own rows (no merge with the file)."""

    def _write(self, new):
        self.rows.update(new)
        write_json(self.path, self.rows)


def test_writes_merge_the_file(tmp_path):
    check_both_kept(tmp_path / "t.json", tuner.TuneTable)


def test_negative_overwrite_loses_rows(tmp_path):
    must_fail(
        lambda: check_both_kept(tmp_path / "t.json", OverwritingTable),
        "a second process's rows dropped",
    )
