"""vswhere runs at most once per process and not again in a later process while the
Visual Studio installs it listed still exist (`<cache>/toolchain/vswhere.json`)."""

from types import SimpleNamespace

import pytest

from kohakumoss_kernels.hardware import toolchain
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _setup(monkeypatch, tmp_path) -> tuple[list, object]:
    exe = tmp_path / "vswhere.exe"
    exe.write_text("")
    install = tmp_path / "VS"
    install.mkdir()
    calls: list = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=f"{install}\n")

    monkeypatch.setattr(toolchain, "VSWHERE", exe)
    monkeypatch.setattr(toolchain, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(toolchain.subprocess, "run", run)
    toolchain._vswhere.cache_clear()
    return calls, install


def check_runs(vswhere, calls: list) -> None:
    assert vswhere("-all") == vswhere("-all")
    assert len(calls) == 1, f"vswhere ran {len(calls)} times in one process"
    getattr(vswhere, "cache_clear", lambda: None)()
    vswhere("-all")
    assert len(calls) == 1, "vswhere ran again in a new process"


def test_vswhere_once_per_process_and_cached_on_disk(monkeypatch, tmp_path):
    calls, install = _setup(monkeypatch, tmp_path)
    check_runs(toolchain._vswhere, calls)
    install.rmdir()
    toolchain._vswhere.cache_clear()
    toolchain._vswhere("-all")
    assert len(calls) == 2, "a vanished install did not rerun vswhere"
    toolchain._vswhere.cache_clear()


def test_negative_uncached_vswhere(monkeypatch, tmp_path):
    calls, _ = _setup(monkeypatch, tmp_path)

    def uncached(*args):
        out = toolchain.subprocess.run([str(toolchain.VSWHERE), *args]).stdout
        return [line for line in out.splitlines() if line]

    must_fail(lambda: check_runs(uncached, calls), "vswhere per call")
    toolchain._vswhere.cache_clear()
