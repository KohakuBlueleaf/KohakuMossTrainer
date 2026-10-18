"""MFU peaks: the card name resolves to its nominal tensor-core peak (the first table
key it contains, so "H100 PCIe" wins over "H100"), unknown cards and non-CUDA devices
to None; on a CUDA device the visible card's peak is known."""

import pytest
import torch

from kohakumoss.model import flops
from kohakumoss.model.flops import PEAK_TFLOPS, peak_tflops
from tests._util.negative import must_fail

CASES = {
    "NVIDIA GeForce RTX 4090": 165.2,
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA H100 80GB HBM3": 989.0,
    "Tesla V100-SXM2-16GB": 125.0,
    "NVIDIA GeForce RTX 3060": 25.6,
    "Tesla T10": 68.8,
    "Some Future GPU": None,
}


def check_peaks(peak, monkeypatch) -> None:
    for name, want in CASES.items():
        monkeypatch.setattr(flops.torch.cuda, "get_device_name", lambda d, n=name: n)
        assert peak(torch.device("cuda")) == want, (name, want)
    assert peak(torch.device("cpu")) is None


def _last_match(device):
    name = torch.cuda.get_device_name(device)
    return next((v for k, v in reversed(PEAK_TFLOPS) if k in name), None)


@pytest.mark.cpu
def test_peak_by_name(monkeypatch):
    check_peaks(peak_tflops, monkeypatch)
    must_fail(lambda: check_peaks(_last_match, monkeypatch), "the last key matched")


@pytest.mark.cuda
def test_peak_known():
    name = torch.cuda.get_device_name(torch.device("cuda"))
    want = next((v for k, v in PEAK_TFLOPS if k in name), None)
    assert want is not None, f"no PEAK_TFLOPS entry for {name}"
    assert peak_tflops(torch.device("cuda")) == want > 0, name
