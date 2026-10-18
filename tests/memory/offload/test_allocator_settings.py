"""configure_allocator: expandable segments on Linux, power-of-2 rounding of request sizes
where the CUDA allocator has no expandable segments (Windows), and an allocator setting
in the environment wins over both (model-free: the setter is recorded, not applied)."""

import pytest

from kohakumoss.offload import accel
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _configured(monkeypatch, configure, platform: str, env: str | None = None):
    applied: list[str] = []
    monkeypatch.setattr(accel, "_SET_ALLOCATOR", applied.append)
    monkeypatch.setattr(accel.sys, "platform", platform)
    for name in accel.ALLOC_ENV:
        monkeypatch.delenv(name, raising=False)
    if env is not None:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", env)
    return configure("cuda"), applied


def check_settings(monkeypatch, configure) -> None:
    got, applied = _configured(monkeypatch, configure, "linux")
    assert got == accel.EXPANDABLE and applied == [accel.EXPANDABLE], (got, applied)
    got, applied = _configured(monkeypatch, configure, "win32")
    assert got == accel.ROUNDED and applied == [accel.ROUNDED], (got, applied)
    got, applied = _configured(monkeypatch, configure, "win32", "max_split_size_mb:64")
    assert applied == [] and "max_split_size_mb:64" in got, (got, applied)
    assert configure("cpu") == "cpu default"


def _defaults_off_linux(device) -> str:
    """The variant that leaves torch's defaults where there are no expandable segments."""
    if accel.sys.platform != "linux":
        return "default"
    return accel.configure_allocator(device)


def test_allocator_settings_per_platform(monkeypatch):
    check_settings(monkeypatch, accel.configure_allocator)
    must_fail(
        lambda: check_settings(monkeypatch, _defaults_off_linux),
        "torch's defaults on Windows",
    )
