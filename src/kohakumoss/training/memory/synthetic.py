"""Synthetic MOSS-shaped block stack for memory / parallel tests and benches.

Same interface as the real backbone pieces: blocks take `(x, BlockCtx)` and expose
`sub_blocks()`; the model exposes `prologue`, `layers`, `epilogue`.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.parallel.parts import PartsCtx
from kohakumoss.training.memory.runners import PlainRunner
from kohakumoss.types.api import BlockCtx
from kohakumoss.types.seqinfo import SeqInfo, build_seqinfo


def _acc(x: Tensor) -> torch.dtype:
    return torch.promote_types(x.dtype, torch.float32)


def _norm_mod(x: Tensor, shift: Tensor, scale: Tensor, seg: Tensor) -> Tensor:
    xf = x.to(_acc(x))
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    y = y * (1 + scale.index_select(0, seg)) + shift.index_select(0, seg)
    return y.to(x.dtype)


class SyntheticBlock(nn.Module):
    """Modulated attention-free mixer + GLU FFN with per-sample gates."""

    def __init__(self, dim: int, ffn: int, index: int, device=None, dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)
        self.index = index
        self.proj = nn.ModuleDict(
            {
                "q": MossLinear(dim, dim, bias=False, **kw),
                "o": MossLinear(dim, dim, bias=False, **kw),
                "ffn_up_gate": MossLinear(dim, 2 * ffn, bias=False, **kw),
                "ffn_down": MossLinear(ffn, dim, bias=False, **kw),
            }
        )
        for lin in self.proj.values():
            lin.init_weights(dim**-0.5)

    def attn_part(self, x: Tensor, ctx: BlockCtx) -> Tensor:
        seg, mod = ctx.seq.seg_joint, ctx.mod
        y = _norm_mod(x, mod[:, 0], mod[:, 1], seg)
        o = self.proj["o"](F.gelu(self.proj["q"](y)))
        r = ctx.residual_scale[self.index]
        return x * r + (o.to(_acc(o)) * mod[:, 2].index_select(0, seg)).to(x.dtype)

    def ffn_part(self, x: Tensor, ctx: BlockCtx) -> Tensor:
        seg, mod = ctx.seq.seg_joint, ctx.mod
        y = _norm_mod(x, mod[:, 3], mod[:, 4], seg)
        u, g = self.proj["ffn_up_gate"](y).chunk(2, -1)
        o = self.proj["ffn_down"](u * F.silu(g))
        return x + (o.to(_acc(o)) * mod[:, 5].index_select(0, seg)).to(x.dtype)

    def sub_blocks(self):
        return (self.attn_part, self.ffn_part)

    def forward(self, x: Tensor, ctx: BlockCtx) -> Tensor:
        return self.ffn_part(self.attn_part(x, ctx), ctx)


class SyntheticModel(nn.Module):
    """prologue (embed + modulation) -> layers -> epilogue (image-token head)."""

    def __init__(
        self, dim: int = 64, ffn: int = 128, depth: int = 4, chans: int = 8, **kw
    ):
        super().__init__()
        self.dim, self.depth = dim, depth
        self.embed = nn.Linear(chans, dim, **kw)
        self.text = nn.Linear(chans, dim, **kw)
        self.time = nn.Linear(1, 8 * dim, **kw)
        self.layers = nn.ModuleList(
            SyntheticBlock(dim, ffn, i, **kw) for i in range(depth)
        )
        self.out = nn.Linear(dim, chans, **kw)
        self.register_buffer("bands", torch.ones(dim // 8), persistent=False)
        self.residual_scale = (1.0,) * depth
        self.runner = PlainRunner()

    def set_runner(self, runner) -> None:
        self.runner = runner

    def prologue(self, img: Tensor, txt: Tensor, sigma: Tensor, seq: SeqInfo):
        t = sigma[:, None].to(self.time.weight.dtype)
        cond = (self.time(t) * 0.1).unflatten(-1, (8, -1))
        mod, head = cond[:, :6].float(), cond[:, 6:].float()
        x = img.new_empty(seq.n_joint, self.dim)
        x = x.index_copy(0, seq.img_in_joint, self.embed(img))
        x = x.index_copy(0, seq.txt_in_joint, self.text(txt))
        return x, PartsCtx(seq, mod, self.bands, self.residual_scale, head)

    def epilogue(self, x: Tensor, ctx: PartsCtx) -> Tensor:
        xi = x.index_select(0, ctx.seq.img_in_joint)
        scale = ctx.head[:, 1].index_select(0, ctx.seq.seg_img)
        return self.out((xi.to(_acc(xi)) * (1 + scale)).to(xi.dtype))

    def prologue_modules(self) -> list[nn.Module]:
        return [self.embed, self.text, self.time]

    def epilogue_modules(self) -> list[nn.Module]:
        return [self.out]

    def forward(self, img, txt, sigma, seq):
        x, ctx = self.prologue(img, txt, sigma, seq)
        run = self.runner.run
        for i, block in enumerate(self.layers):
            x = run(i, block, x, ctx)
        return self.epilogue(x, ctx)


def synthetic_batch(
    hw: list[tuple[int, int]],
    text: list[int],
    chans: int = 8,
    device="cpu",
    dtype=torch.float32,
    seed: int = 0,
):
    """(img, txt, sigma, seq, target) for a packed batch of the given sizes."""
    g = torch.Generator().manual_seed(seed)
    seq = build_seqinfo(torch.tensor(hw), torch.tensor(text)).to(device)
    img = torch.randn(seq.n_img, chans, generator=g).to(device, dtype)
    txt = torch.randn(seq.n_txt, chans, generator=g).to(device, dtype)
    sigma = torch.rand(len(hw), generator=g).to(device)
    target = torch.randn(seq.n_img, chans, generator=g).to(device, dtype)
    return img, txt, sigma, seq, target
