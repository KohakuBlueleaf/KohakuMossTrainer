"""SSIM and PSNR of the probe scripts: SSIM against its closed form on a pair of constant
images, 1 on equal images, falling with the noise, about 0 against a pixel shuffle; PSNR
against the mean squared error at peak-to-peak 2."""

import math

import pytest
import torch

from kohakumoss.probe import image_metrics
from kohakumoss.probe.image_metrics import psnr, ssim
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

C1, C2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2


def _constant_pair(fn) -> None:
    a, b = torch.full((1, 3, 32, 32), 0.3), torch.full((1, 3, 32, 32), -0.2)
    want = (2 * 0.3 * -0.2 + C1) / (0.3**2 + 0.2**2 + C1)
    assert abs(fn(a, b) - want) < 1e-4


def _structure(fn) -> None:
    g = torch.Generator().manual_seed(0)
    a = torch.rand((1, 3, 64, 64), generator=g) * 2 - 1
    assert abs(fn(a, a) - 1.0) < 1e-6
    noisy = [
        fn(a, (a + s * torch.randn(a.shape, generator=g)).clamp(-1, 1))
        for s in (0.05, 0.2, 0.5)
    ]
    assert noisy[0] > noisy[1] > noisy[2]
    shuffled = a.flatten(2)[..., torch.randperm(a[0, 0].numel(), generator=g)]
    assert abs(fn(a, shuffled.view_as(a))) < 0.05


def test_ssim_constant_closed_form():
    _constant_pair(ssim)


def test_ssim_structure():
    _structure(ssim)


def test_ssim_negative_unnormalised_window(monkeypatch):
    monkeypatch.setattr(image_metrics, "WINDOW", image_metrics.WINDOW * 2)
    must_fail(lambda: _constant_pair(image_metrics.ssim), "window not summing to 1")


def test_ssim_negative_structure_lost():
    def flat(a, b):
        return ssim(a, torch.full_like(b, float(b.mean())))

    must_fail(lambda: _structure(flat), "second image replaced by its mean")


def _psnr_at(a, b, peak: float) -> None:
    mse = float((a - b).pow(2).mean())
    assert abs(10 * math.log10(peak**2 / mse) - psnr(a, b)) < 1e-9


def test_psnr_matches_mse():
    a = torch.zeros((1, 3, 8, 8))
    b = torch.full((1, 3, 8, 8), 0.2)
    _psnr_at(a, b, 2.0)
    assert psnr(a, a) == 99.0
    must_fail(lambda: _psnr_at(a, b, 1.0), "peak-to-peak 1 instead of 2")
