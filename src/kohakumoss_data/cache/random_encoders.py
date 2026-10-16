"""Deterministic stand-in encoders with the EncodeImages / EncodeText signatures.

Used by tests, smoke runs and cache-pipeline checks when no real VAE / text encoder is
loaded. `RandomImageEncoder` averages 16x16 patches and projects them to C channels;
`RandomTextEncoder` embeds UTF-8 bytes (+ a template of `overhead` tokens).
"""

import numpy as np
import torch
from torch import Tensor


class RandomImageEncoder:
    """uint8 [H, W, 3] images -> packed [sum(h*w), C] (patch means x fixed projection)."""

    def __init__(
        self, channels: int = 128, factor: int = 16, seed: int = 0, device="cpu"
    ):
        gen = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(3, channels, generator=gen).to(device)
        self.factor = factor
        self.device = device

    def __call__(self, images, hw: Tensor) -> Tensor:
        out = []
        f = self.factor
        for img, (h, w) in zip(images, hw.tolist()):
            x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).float()
            x = x[: h * f, : w * f].reshape(h, f, w, f, 3).mean((1, 3)) / 127.5 - 1.0
            out.append((x.reshape(h * w, 3) @ self.proj).to(torch.bfloat16))
        return torch.cat(out)


class RandomTextEncoder:
    """prompts -> [overhead + len(utf8), D] embeddings from a fixed byte table."""

    def __init__(
        self, dim: int = 4096, overhead: int = 12, seed: int = 0, device="cpu"
    ):
        gen = torch.Generator().manual_seed(seed)
        self.table = torch.randn(256 + overhead, dim, generator=gen).to(device)
        self.overhead = overhead
        self.device = device

    def __call__(self, prompts) -> list[Tensor]:
        out = []
        for prompt in prompts:
            ids = list(range(256, 256 + self.overhead)) + list(prompt.encode("utf-8"))
            index = torch.tensor(ids, device=self.device)
            out.append(self.table[index].to(torch.bfloat16))
        return out
