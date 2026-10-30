"""The doctor's missing-requirement report: extras expand through self-references,
markers select per extra and platform, unmet requirements carry their fix command."""

import importlib.metadata as metadata

import pytest
from packaging import markers
from packaging.requirements import Requirement
from packaging.version import Version

from kohakumoss.cli import missing
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def names(reqs) -> set[str]:
    return {r.name for r in reqs}


def test_default_extras_expand():
    cuda = names(missing.requirements("cuda"))
    assert {"bitsandbytes", "tilelang"} <= cuda, cuda
    assert "kohakumoss" not in cuda
    xpu = names(missing.requirements("xpu"))
    assert "bitsandbytes" in xpu and "tilelang" not in xpu, xpu
    base = names(missing.requirements(None))
    assert {"torch", "kohaku-engine", "lycoris-lora"} <= base, base
    assert "tilelang" not in base, base


@pytest.mark.parametrize(
    "text, extra, platform, applies",
    [
        ("a", None, "linux", True),
        ("a", "kernels", "linux", False),
        ('a; extra == "kernels"', "kernels", "linux", True),
        ('a; extra == "kernels"', None, "linux", False),
        ('a; sys_platform == "win32" and extra == "kernels"', "kernels", "win32", True),
        (
            'a; sys_platform == "win32" and extra == "kernels"',
            "kernels",
            "linux",
            False,
        ),
        ('a; sys_platform != "win32"', None, "linux", True),
        ('a; sys_platform != "win32"', "kernels", "linux", False),
    ],
)
def test_marker_selection(text, extra, platform, applies, monkeypatch):
    env = {**markers.default_environment(), "sys_platform": platform}
    monkeypatch.setattr(markers, "default_environment", lambda: dict(env))
    assert missing._applies(Requirement(text), extra) is applies


def test_unmet():
    assert missing.unmet(Requirement("kohakumoss-no-such-package")) is not None
    assert missing.unmet(Requirement("torch")) is None
    installed = Version(metadata.version("torch")).base_version
    assert "does not satisfy" in missing.unmet(Requirement(f"torch<{installed}"))
    must_fail(
        lambda: _assert_none(missing.unmet(Requirement("kohakumoss-no-such-package"))),
        "absent package reported satisfied",
    )


def _assert_none(value):
    assert value is None


def test_missing_rows_name_the_fix(monkeypatch):
    fake = ["kohakumoss-no-such-package>=1", 'kohakumoss-gone; extra == "cuda"']
    real = metadata.requires

    def requires(name):
        return fake if name == missing.PACKAGE else real(name)

    monkeypatch.setattr(missing.metadata, "requires", requires)
    monkeypatch.setattr(missing, "accelerator_kind", lambda: "cuda")
    monkeypatch.setattr(missing, "_torch_rows", lambda: [])
    monkeypatch.setattr(missing, "_toolchain_rows", lambda kind, probes: [])
    monkeypatch.setattr(missing, "_msvc_rows", lambda msvc, kind: [])
    rows = missing.missing({"needed": False})
    fixes = {r["what"].split()[0]: r["fix"] for r in rows}
    assert '-e "."' in fixes["kohakumoss-no-such-package"]
    assert '-e ".[cuda]"' in fixes["kohakumoss-gone"]
    index = f"download.pytorch.org/whl/{missing.index_tag('cuda')}"
    assert all(index in f and "--torch-backend" not in f for f in fixes.values())


def test_backend_rows():
    probes = {
        "tilelang@cuda:0": "RuntimeError: nvcc not found",
        "triton@cuda:0": "cl.exe failed",
        "eager@cpu": None,
    }
    rows = {r["what"].split()[1]: r["fix"] for r in missing.backend_rows(probes)}
    assert set(rows) == {"tilelang@cuda:0", "triton@cuda:0"}
    assert rows["tilelang@cuda:0"] == missing.NVCC_FIX
    assert "MSVC" in rows["triton@cuda:0"]
    must_fail(lambda: _assert_none(missing.backend_rows(probes) or None), "no rows")


def test_nvcc_lookup(monkeypatch, tmp_path):
    monkeypatch.setattr(missing.shutil, "which", lambda name: None)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert not missing.nvcc()
    exe = "nvcc.exe" if missing.sys.platform == "win32" else "nvcc"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / exe).write_bytes(b"")
    assert missing.nvcc()


def test_nvcc_row_only_without_a_working_tilelang(monkeypatch):
    monkeypatch.setattr(missing, "nvcc", lambda: False)
    monkeypatch.setattr(missing.sys, "platform", "linux")
    assert missing._toolchain_rows("cuda", {"tilelang@cuda:0": "no nvcc"})
    assert not missing._toolchain_rows("cuda", {"tilelang@cuda:0": None})
    assert not missing._toolchain_rows("cpu", {})


def test_xpu_rows(monkeypatch):
    monkeypatch.setattr(missing, "unmet", lambda req: None)
    rows = missing._xpu_rows("xpu")
    assert [r["fix"] for r in rows] == ["uv pip uninstall triton-windows"]
    assert missing._xpu_rows("cuda") == []


def test_msvc_row(monkeypatch):
    monkeypatch.setattr(missing.sys, "platform", "win32")
    absent = {"needed": True, "cl_on_path": False, "vswhere_sees_vs": False}
    assert missing._msvc_rows({**absent, "msvc_versions": []}, "cuda")
    assert not missing._msvc_rows({**absent, "msvc_versions": ["14.51"]}, "cuda")
    assert not missing._msvc_rows({**absent, "msvc_versions": []}, "cpu")
