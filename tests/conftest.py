"""Test configuration: hardware-marker skipping and deterministic seeds."""

import os
import sys

import pytest
import torch

from kohakumoss_kernels.hardware.arch import arch_of

SM_MARKERS = {"sm70": 70, "sm75": 75, "sm86": 86, "sm89": 89, "sm120": 120}


def _cuda_arch():
    if not torch.cuda.is_available():
        return None
    return arch_of("cuda")


def pytest_collection_modifyitems(config, items):
    arch = _cuda_arch()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for item in items:
        marks = {m.name for m in item.iter_markers()}
        if "cuda" in marks and arch is None:
            item.add_marker(pytest.mark.skip(reason="no CUDA GPU"))
        for name, sm in SM_MARKERS.items():
            if name in marks and (arch is None or arch.sm != sm):
                item.add_marker(pytest.mark.skip(reason=f"needs {name}"))
        if "xpu" in marks and not has_xpu:
            item.add_marker(pytest.mark.skip(reason="no XPU"))
        if "windows" in marks and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(reason="Windows only"))
        if "multi_gpu" in marks and n_gpu < 2:
            item.add_marker(pytest.mark.skip(reason="needs two GPUs"))
        if "real_weights" in marks and not os.environ.get("KOHAKUMOSS_WEIGHTS"):
            item.add_marker(pytest.mark.skip(reason="KOHAKUMOSS_WEIGHTS unset"))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


@pytest.fixture
def weights_dir():
    return os.environ.get("KOHAKUMOSS_WEIGHTS")
