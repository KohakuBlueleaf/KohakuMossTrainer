"""FLUX.2 VAE wrapper: pixels <-> normalised 128-channel /16 latents.

Encode: VAE posterior (mode or sample) [B, 32, H/8, W/8] -> 2x2 patchify -> 128 ch ->
(z - running_mean) / sqrt(running_var + eps). Decode is the exact inverse.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from torch import Tensor, nn


def patchify(z: Tensor) -> Tensor:
    """[B, C, 2h, 2w] -> [B, 4C, h, w]; channel = c * 4 + i * 2 + j for pixel (2y+i, 2x+j)."""
    b, c, hh, ww = z.shape
    z = z.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return z.reshape(b, c * 4, hh // 2, ww // 2)


def unpatchify(z: Tensor) -> Tensor:
    """Inverse of `patchify`."""
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(b, c // 4, 2 * h, 2 * w)


def to_pixels(image: np.ndarray, device, dtype) -> Tensor:
    """uint8 HWC (RGB) -> [1, 3, H, W] in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    return x.permute(2, 0, 1).unsqueeze(0).to(dtype) / 127.5 - 1.0


class FluxVae(nn.Module):
    """Frozen VAE with the MOSS latent normalisation; `mode` is "mode" or "sample"."""

    def __init__(self, vae: AutoencoderKLFlux2, mode: str = "mode"):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        bn = vae.bn
        eps = vae.config.batch_norm_eps
        self.register_buffer("mean", bn.running_mean.detach().clone().float(), False)
        self.register_buffer(
            "std", (bn.running_var.detach().float() + eps).sqrt(), False
        )
        match mode:
            case "mode":
                self._draw = _mode
            case "sample":
                self._draw = _sample
            case _:
                raise ValueError(f"unknown VAE encode mode {mode!r}")

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        mode: str = "mode",
    ) -> "FluxVae":
        """`path` is the VAE dir or a pipeline root containing `vae/`."""
        path = Path(path)
        sub = "vae" if (path / "vae" / "config.json").is_file() else None
        vae = AutoencoderKLFlux2.from_pretrained(
            str(path), subfolder=sub, torch_dtype=dtype
        )
        return cls(vae, mode).to(device)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @torch.no_grad()
    def encode(
        self, pixels: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        """pixels [B, 3, H, W] in [-1, 1] (H, W multiples of 16) -> fp32 [B, 128, H/16, W/16]."""
        dist = self.vae.encode(pixels.to(self.dtype)).latent_dist
        z = patchify(self._draw(dist, generator).float())
        return (z - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """fp32 [B, 128, h, w] -> pixels [B, 3, 16h, 16w] in [-1, 1]."""
        z = latents.float() * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)
        out = self.vae.decode(unpatchify(z).to(self.dtype), return_dict=False)[0]
        return out.float().clamp(-1, 1)

    def encode_images(
        self, images: Sequence[np.ndarray], hw: Tensor, generator=None
    ) -> Tensor:
        """EncodeImages: uint8 HWC images sized 16*hw -> packed fp32 latents [N_img, 128]."""
        out = []
        device = self.mean.device
        for image, (h, w) in zip(images, hw.tolist()):
            if image.shape[0] != 16 * h or image.shape[1] != 16 * w:
                raise ValueError(f"image {image.shape[:2]} is not 16 * {(h, w)}")
            z = self.encode(to_pixels(image, device, self.dtype), generator)
            out.append(z[0].flatten(1).transpose(0, 1))
        return torch.cat(out)


def _mode(dist, generator):
    return dist.mode()


def _sample(dist, generator):
    return dist.sample(generator)
