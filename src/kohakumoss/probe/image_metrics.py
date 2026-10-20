"""Image distances of the probe scripts: SSIM and PSNR of [1, 3, H, W] images in [-1, 1]."""

import math

import torch
import torch.nn.functional as F


def _window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


WINDOW = _window()


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM (11x11 Gaussian window, sigma 1.5, data range 2, per channel, valid
    region) of two [1, 3, H, W] images in [-1, 1]."""
    a, b = a.float().cpu(), b.float().cpu()
    w = WINDOW.expand(a.shape[1], 1, -1, -1)
    g = a.shape[1]
    c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    mu_a, mu_b = F.conv2d(a, w, groups=g), F.conv2d(b, w, groups=g)
    saa = F.conv2d(a * a, w, groups=g) - mu_a**2
    sbb = F.conv2d(b * b, w, groups=g) - mu_b**2
    sab = F.conv2d(a * b, w, groups=g) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2)
    return float((num / den).mean())


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB of two [-1, 1] images (peak-to-peak 2); 99 when equal."""
    mse = float(((a.float() - b.float()) / 2).pow(2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)
