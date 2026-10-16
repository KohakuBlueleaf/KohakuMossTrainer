"""Independent restatement of the official MOSS-Image transformer forward (Apache-2.0
reference pipeline), without diffusers base classes, used as the test oracle."""

import torch
import torch.nn.functional as F
from torch import nn


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
    return out.transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads, n_blocks):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList(
            [_TextBlock(dim, ffn_dim, heads) for _ in range(n_blocks)]
        )

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {n: nn.Linear(dim, dim, bias=False) for n in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        y = y.to(self.proj["q"].weight.dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        v = self.proj["v"](y)
        x = x + self.proj["o"](_attention(q, k, v, rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class OfficialTransformer(nn.Module):
    """Same keys as the HF checkpoint; forward over padded [B, C, H, W] + text mask."""

    def __init__(self, cfg):
        super().__init__()
        dim, heads = cfg.dim, cfg.heads
        self.cfg = cfg
        self.video_embedder = nn.Linear(cfg.latent_channels, dim)
        self.text_embedder = _TextProjection(
            cfg.text_dim, dim, cfg.ffn_dim, heads, cfg.text_blocks
        )
        nb = (dim // heads) // 8
        bands = 1 / (cfg.rope_base ** (torch.arange(nb, dtype=torch.float32) / nb))
        self.register_buffer("rope_bands", bands, persistent=False)
        self.time_embedding = nn.Sequential(
            nn.Linear(cfg.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [_Block(dim, cfg.ffn_dim, heads) for _ in range(cfg.depth)]
        )
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, cfg.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def rope(self, height, width, text_length, device):
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.rope_bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)

    def forward_one(self, latent, timestep, context, skip_layers=()):
        """latent [1, C, H, W], timestep [1] in [0, 1000], context [1, L, text_dim]."""
        _, _, height, width = latent.shape
        half = self.cfg.freq_dim // 2
        freq = 10000 ** (
            -torch.arange(half, device=latent.device, dtype=torch.float64) / half
        )
        phase = timestep.double()[:, None] * freq
        emb = torch.cat((phase.cos(), phase.sin()), -1).float()
        dt = self.time_embedding[0].weight.dtype
        time = self.time_embedding(emb.to(dt)).float()
        mod = self.shared_block_modulation(time.to(dt)).view(1, 6, -1)
        head = self.video_head_modulation(time.to(dt)).view(1, 2, -1)
        video = latent.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype))
        rope = self.rope(height, width, context.shape[1], latent.device).to(video.dtype)
        text = self.text_embedder(context, rope[:, height * width :])
        x = torch.cat((video, text), 1)
        for i, layer in enumerate(self.layers):
            if i not in skip_layers:
                x = layer(x, rope, mod)
        x = self.final_norm_video(x[:, : height * width])
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x)
        return x.transpose(1, 2).reshape(1, -1, height, width)
