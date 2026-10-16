"""A small model with MOSS module paths, built from MossLinear, for adapter tests."""

import torch
import torch.nn.functional as F
from torch import nn

from kohakumoss.model.linear import MossLinear

D, FFN, TEXT, LAT = 32, 48, 24, 8


class Norm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


def _lin(i, o, bias=False):
    m = MossLinear(i, o, bias=bias)
    m.init_weights(0.2)
    return m


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = Norm(D), Norm(D)
        self.q_norm, self.k_norm = Norm(D), Norm(D)
        self.proj = nn.ModuleDict(
            {
                "q": _lin(D, D),
                "k": _lin(D, D),
                "v": _lin(D, D),
                "o": _lin(D, D),
                "ffn_up_gate": _lin(D, 2 * FFN),
                "ffn_down": _lin(FFN, D),
            }
        )

    def forward(self, x):
        p = self.proj
        y = self.norm1(x)
        a = self.q_norm(p["q"](y)) * torch.sigmoid(self.k_norm(p["k"](y))) + p["v"](y)
        x = x + p["o"](a)
        u, g = p["ffn_up_gate"](self.norm2(x)).chunk(2, -1)
        return x + p["ffn_down"](u * torch.sigmoid(g))


class TextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = Norm(D), Norm(D)
        self.qkv = _lin(D, 3 * D, bias=True)
        self.proj = _lin(D, D, bias=True)
        self.ffn_gate = _lin(D, FFN, bias=True)
        self.ffn_up = _lin(D, FFN, bias=True)
        self.ffn_down = _lin(FFN, D, bias=True)

    def forward(self, x):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        x = x + self.proj(q * torch.tanh(k) + v)
        y = self.norm2(x)
        return x + self.ffn_down(self.ffn_up(y) * torch.sigmoid(self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _lin(TEXT, D, bias=True)
        self.blocks = nn.ModuleList([TextBlock() for _ in range(2)])

    def forward(self, t):
        t = self.proj(t)
        for b in self.blocks:
            t = b(t)
        return t


class Toy(nn.Module):
    """forward(latent [N, LAT], text [N, TEXT]) -> [N, LAT]; every linear on the path."""

    def __init__(self, depth: int = 2):
        super().__init__()
        self.video_embedder = _lin(LAT, D, bias=True)
        self.text_embedder = TextEmbedder()
        self.time_embedding = nn.Sequential(
            _lin(D, D, True), nn.SiLU(), _lin(D, D, True)
        )
        self.shared_block_modulation = _lin(D, 2 * D, bias=True)
        self.layers = nn.ModuleList([Block() for _ in range(depth)])
        self.final_norm_video = Norm(D)
        self.final_linear_video = _lin(D, LAT)

    def forward(self, latent, text):
        h = self.video_embedder(latent) + self.text_embedder(text)
        shift, scale = self.shared_block_modulation(self.time_embedding(h)).chunk(2, -1)
        h = h * (1 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        for layer in self.layers:
            h = layer(h)
        return self.final_linear_video(self.final_norm_video(h))


def toy(seed: int = 0, depth: int = 2, dtype=torch.float32) -> Toy:
    torch.manual_seed(seed)
    model = Toy(depth).to(dtype)
    model.requires_grad_(False)
    return model


def inputs(n: int = 7, seed: int = 1, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, LAT, generator=g).to(dtype),
        torch.randn(n, TEXT, generator=g).to(dtype),
    )


def perturb(aset, seed: int = 2, std: float = 0.1) -> None:
    """Every trainable adapter tensor random (LyCORIS zero-inits one factor)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in aset.named_trainable():
            noise = torch.randn(p.shape, generator=g).to(p) * std
            if any(tag in name for tag in ("dora_scale", "rescale", "lambda_layer")):
                p.mul_(1 + noise)
            else:
                p.add_(noise)
