"""Deterministic encode functions with the EncodeImages / EncodeText signatures for the
model-free data and cache tests (no model, no random weights): `PatchMeans` maps each
16x16 pixel patch to its channel means spread over C channels by a fixed formula,
`ByteCodes` maps each UTF-8 byte (after `overhead` template codes) to a fixed vector."""

import numpy as np
import torch
from torch import Tensor


def _channel_map(channels: int, device) -> Tensor:
    """[3, C]: channel c reads RGB channel c % 3, scaled by 1 + c // 3 / C."""
    out = torch.zeros(3, channels, device=device)
    c = torch.arange(channels, device=device)
    out[c % 3, c] = 1.0 + (c // 3).float() / channels
    return out


class PatchMeans:
    """uint8 [H, W, 3] images -> packed bf16 [sum(h*w), C]: each patch's mean in
    [-1, 1], channel c the mean of RGB channel c % 3 times 1 + c // 3 / C."""

    def __init__(self, channels: int = 128, factor: int = 16, device="cpu"):
        self.map = _channel_map(channels, device)
        self.factor = factor
        self.device = device

    def __call__(self, images, hw: Tensor) -> Tensor:
        out = []
        f = self.factor
        for img, (h, w) in zip(images, hw.tolist()):
            x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).float()
            x = x[: h * f, : w * f].reshape(h, f, w, f, 3).mean((1, 3)) / 127.5 - 1.0
            out.append((x.reshape(h * w, 3) @ self.map).to(torch.bfloat16))
        return torch.cat(out)


def _code_table(rows: int, dim: int, device) -> Tensor:
    """[rows, dim]: row i, column d is cos(i * (d + 1) / dim)."""
    i = torch.arange(rows, device=device, dtype=torch.float32)[:, None]
    d = torch.arange(dim, device=device, dtype=torch.float32)[None, :]
    return torch.cos(i * (d + 1) / dim)


class ByteCodes:
    """prompts -> bf16 [overhead + len(utf8), D]: `overhead` template codes, then one
    code per UTF-8 byte (row 256 + k for template position k, row b for byte b)."""

    encoder_id = "byte-codes"

    def __init__(self, dim: int = 4096, overhead: int = 12, device="cpu"):
        self.table = _code_table(256 + overhead, dim, device)
        self.overhead = overhead
        self.device = device

    def __call__(self, prompts) -> list[Tensor]:
        out = []
        for prompt in prompts:
            ids = list(range(256, 256 + self.overhead)) + list(prompt.encode("utf-8"))
            index = torch.tensor(ids, device=self.device)
            out.append(self.table[index].to(torch.bfloat16))
        return out
