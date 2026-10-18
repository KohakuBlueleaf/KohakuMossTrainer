"""The Windows liveness check declares its kernel32 prototypes: OpenProcess returns a
pointer-sized HANDLE (the default int restype truncates it on 64-bit Windows)."""

import ctypes
import os
import sys

import pytest

from kohakumoss_data.scan import lock
from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(sys.platform != "win32", reason="kernel32 exists on Windows"),
]


def check_prototypes(kernel) -> None:
    wt = ctypes.wintypes
    assert kernel.OpenProcess.restype is wt.HANDLE
    assert kernel.OpenProcess.argtypes == (wt.DWORD, wt.BOOL, wt.DWORD)
    assert kernel.GetExitCodeProcess.restype is wt.BOOL
    assert kernel.CloseHandle.argtypes == (wt.HANDLE,)


def test_kernel32_prototypes_and_liveness():
    check_prototypes(lock.KERNEL32)
    assert lock.pid_alive(os.getpid())
    assert not lock.pid_alive(2**31 - 7)


def test_negative_undeclared_prototypes():
    must_fail(lambda: check_prototypes(ctypes.WinDLL("kernel32")), "default prototypes")
