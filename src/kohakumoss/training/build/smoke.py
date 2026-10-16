"""Deterministic stand-in encoders for WEIGHTS='random' smoke runs (no downloads).

They satisfy EncodeImages / EncodeText and the decode / empty calls of the pipeline;
outputs carry no meaning beyond being fixed functions of their inputs.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


class RandomLatentEncoder:
    """Area-downsample to the latent grid, then a fixed 3 -> C projection."""

    def __init__(self, channels: int, factor: int = 16, seed: int = 0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(channels, 3, generator=g).to(device)
        self.factor = factor
        self.device = device

    def to(self, device) -> "RandomLatentEncoder":
        """Placement hook of the pipeline residency; the projection follows `device`."""
        self.proj = self.proj.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def encode_images(self, images: Sequence[np.ndarray], hw: Tensor, generator=None) -> Tensor:
        out = []
        for image, (h, w) in zip(images, hw.tolist()):
            x = torch.from_numpy(np.ascontiguousarray(image[..., :3])).float().to(self.device)
            x = x.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            z = F.adaptive_avg_pool2d(x, (int(h), int(w)))[0]
            out.append((self.proj @ z.flatten(1)).T)
        return torch.cat(out, 0)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        rgb = torch.einsum(
            "ck,bchw->bkhw", self.proj, latents.float().to(self.proj.device)
        ) / self.proj.shape[0]
        return F.interpolate(rgb.tanh(), scale_factor=self.factor, mode="nearest")


class RandomTextEncoder:
    """Comma-separated tokens (+ 2 template tokens), each a hash-seeded vector."""

    def __init__(self, dim: int, device="cpu"):
        self.dim = dim
        self.device = device

    def _vec(self, token: str) -> Tensor:
        g = torch.Generator().manual_seed(_seed(token))
        return torch.randn(self.dim, generator=g)

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        out = []
        for prompt in prompts:
            tokens = ["<bos>"] + [t.strip() for t in prompt.split(",") if t.strip()] + ["<eos>"]
            out.append(torch.stack([self._vec(t) for t in tokens]).to(self.device))
        return out

    def __call__(self, prompts: Sequence[str]) -> list[Tensor]:
        return self.encode(prompts)

    def empty(self) -> Tensor:
        return self.encode([""])[0]
