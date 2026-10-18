"""JSON writes (the one atomic writer, `kohakumoss_common.io.write_json_atomic`, the
kernel cache, the data cache settings and the scan lock all use it): a temporary name
per process, thread and write; concurrent writers never leave a partial or temporary
file; a replace retried while another process holds the file (Windows); a tuning-table
write that fails keeps the choice in memory and never reaches the caller."""

import json
import threading
from pathlib import Path

import pytest
import torch

from kohakumoss_common import io as cio
from kohakumoss_kernels.hardware import device
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")


def _pid_only(path: Path) -> Path:
    return path.with_name(path.name + ".1234.tmp")


def check_unique_names(name_of) -> None:
    """Two writes from one thread and one from another thread get three names."""
    path = Path("table.json")
    names = [name_of(path), name_of(path)]
    other = threading.Thread(target=lambda: names.append(name_of(path)))
    other.start()
    other.join()
    assert len(set(names)) == 3, names


def test_temp_names_are_unique():
    check_unique_names(cio.temp_name)


def test_negative_pid_only_temp_name():
    must_fail(lambda: check_unique_names(_pid_only), "one name per process")


def check_concurrent(tmp_path) -> None:
    """Eight threads writing one file while a reader polls it: every write lands, every
    read that opens the file parses (on Windows an open can be refused while a replace
    is in flight, which the readers' `read_json` allows), no temporary file is left."""
    path, errors, done = tmp_path / "settings.json", [], threading.Event()

    def writer(i):
        for j in range(40):
            try:
                cio.write_json_atomic(path, {"writer": i, "n": j, "pad": "x" * 2000})
            except OSError as err:
                errors.append(err)

    def reader():
        while not done.is_set():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError):
                pass
            except ValueError as err:
                errors.append(err)
            done.wait(0.002)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    polling = threading.Thread(target=reader)
    polling.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    done.set()
    polling.join()
    assert not errors, errors[:3]
    assert json.loads(path.read_text(encoding="utf-8"))["n"] == 39
    assert not list(tmp_path.glob("*.tmp")), "a temporary file was left behind"


def test_concurrent_writers_never_partial(tmp_path):
    check_concurrent(tmp_path)


def test_negative_concurrent_pid_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "temp_name", _pid_only)
    must_fail(lambda: check_concurrent(tmp_path), "pid-only temporary name")


def check_replace_retried(tmp_path, monkeypatch) -> None:
    """A replace refused twice (the file held open elsewhere) still lands the table."""
    real, refused = cio.os.replace, []

    def held(src, dst):
        if len(refused) < 2:
            refused.append(dst)
            raise PermissionError(13, "the file is used by another process")
        real(src, dst)

    monkeypatch.setattr(cio.os, "replace", held)
    monkeypatch.setattr(cio, "REPLACE_WAIT_S", 0.0)
    path = tmp_path / "t.json"
    try:
        device.write_json(path, {"k": 1})
    except PermissionError as err:
        raise AssertionError(f"replace not retried: {err}") from err
    assert device.read_json(path) == {"k": 1}
    assert not list(tmp_path.glob("*.tmp")), "a temporary file was left behind"


def test_replace_retried_on_permission_error(tmp_path, monkeypatch):
    check_replace_retried(tmp_path, monkeypatch)


def test_negative_no_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "REPLACE_TRIES", 1)
    must_fail(lambda: check_replace_retried(tmp_path, monkeypatch), "no retry")


@pytest.fixture
def unwritable(tmp_path, monkeypatch):
    """A kernel cache whose root is a file: every table write raises OSError."""
    root = tmp_path / "cache"
    root.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(root))
    monkeypatch.setattr(tuner, "TIMINGS", {"events": lambda fn, dev: fn()})
    tuner._TABLES.clear()
    tuner.reset()
    tuner.set_mode("table")
    yield
    tuner._TABLES.clear()
    tuner.reset()


def _launch(config):
    return lambda: config["ms"]


def check_write_failure_kept(select) -> None:
    """The measured winner binds for the process although its table cannot be written."""
    configs = [{"ms": 2.0}, {"ms": 1.0}]
    try:
        got = select("test_unwritable", CPU, "k", lambda: configs, _launch)
    except OSError as err:
        raise AssertionError(f"a table write error reached the caller: {err}") from err
    assert got == {"ms": 1.0}, got
    assert tuner.decided("test_unwritable", CPU, "k")
    assert tuner.table("test_unwritable", CPU).get("k") == {"ms": 1.0}


def test_table_write_failure_keeps_the_choice(unwritable):
    check_write_failure_kept(tuner.select)


def test_negative_raising_table_write(unwritable, monkeypatch):
    def raising(self, new):
        self.rows = {**self.rows, **new}
        device.write_json(self.path, self.rows)

    monkeypatch.setattr(tuner.TuneTable, "_write", raising)
    must_fail(lambda: check_write_failure_kept(tuner.select), "write error raised")
