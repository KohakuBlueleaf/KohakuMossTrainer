"""Joint image+text block: shared-modulation pre-norm attention and SiTU FFN with gates."""

from collections.abc import Callable

from torch import Tensor, nn

from kohakumoss.model.components.norm import RMSNorm
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.opset import OpSet
from kohakumoss.model.stream import BlockConfig, caster, clamper, scaled_writer
from kohakumoss.types.arch import MossArchConfig
from kohakumoss.types.context import (
    GATE_ATTN,
    GATE_FFN,
    SCALE_ATTN,
    SCALE_FFN,
    SHIFT_ATTN,
    SHIFT_FFN,
    BlockCtx,
)

PROJ = ("q", "k", "v", "o")


class JointBlock(nn.Module):
    """One of the `depth` joint blocks; `index` selects its residual scale in BlockCtx."""

    def __init__(self, cfg: MossArchConfig, index: int, device=None, dtype=None):
        super().__init__()
        dim, ffn = cfg.dim, cfg.ffn_dim
        kw = dict(device=device, dtype=dtype)
        self.index = index
        self.heads = cfg.heads
        self.norm1 = RMSNorm(dim, cfg.eps, **kw)
        self.norm2 = RMSNorm(dim, cfg.eps, **kw)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.eps, **kw)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.eps, **kw)
        proj = {name: MossLinear(dim, dim, bias=False, **kw) for name in PROJ}
        proj["ffn_up_gate"] = MossLinear(dim, 2 * ffn, bias=False, **kw)
        proj["ffn_down"] = MossLinear(ffn, dim, bias=False, **kw)
        self.proj = nn.ModuleDict(proj)

    def bind(self, ops: OpSet, stream: BlockConfig = BlockConfig()) -> None:
        """Resolve this block's callables once: fused or composed ops, the stream
        config's writer scaling, norm-output cast (fp32 stream) and clamp."""
        p = self.proj
        self.ops = ops
        self._qk = ops.bind_qk_rope(p["q"], p["k"], self.heads)
        self._situ = ops.bind_situ(p["ffn_up_gate"])
        self._attn_out = scaled_writer(
            ops.bind_gated_residual(p["o"]), stream.in_attn, stream.gate_attn
        )
        self._ffn_out = scaled_writer(
            ops.bind_gated_residual(p["ffn_down"]), stream.in_ffn, stream.gate_ffn
        )
        compute = p["q"].weight_like.dtype
        self._cast = caster(compute if stream.stream_dtype is not None else None)
        self._post = clamper(stream.clamp, p["q"].weight_like.device)

    def sub_blocks(self) -> tuple[Callable, Callable]:
        """(attention part, FFN part), each (x, ctx) -> x; forward is their composition."""
        return self.attn_part, self.ffn_part

    def forward(self, x: Tensor, ctx: BlockCtx) -> Tensor:
        return self.ffn_part(self.attn_part(x, ctx), ctx)

    def attn_part(self, x: Tensor, ctx: BlockCtx) -> Tensor:
        """Modulated norm, qk-norm + RoPE, attention, gated residual (with the fold's r)."""
        ops, seq, mod = self.ops, ctx.seq, ctx.mod
        seg = seq.seg_joint
        y = self._cast(
            ops.rms_norm_mod(
                x,
                self.norm1.weight,
                mod[:, SHIFT_ATTN],
                mod[:, SCALE_ATTN],
                seg,
                self.norm1.eps,
            )
        )
        q, k = self._qk(
            y,
            self.q_norm.weight,
            self.k_norm.weight,
            seq.pos_joint,
            ctx.bands,
            self.q_norm.eps,
        )
        v = self.proj["v"](y).unflatten(-1, (self.heads, -1))
        o = ops.attention(q, k, v, seq.attn_joint).flatten(1)
        r = ctx.residual_scale[self.index]
        return self._post(self._attn_out(o, x, mod[:, GATE_ATTN], seg, r))

    def ffn_part(self, x: Tensor, ctx: BlockCtx) -> Tensor:
        """Modulated norm, SiTU FFN, gated residual."""
        ops, mod, seg = self.ops, ctx.mod, ctx.seq.seg_joint
        y = self._cast(
            ops.rms_norm_mod(
                x,
                self.norm2.weight,
                mod[:, SHIFT_FFN],
                mod[:, SCALE_FFN],
                seg,
                self.norm2.eps,
            )
        )
        out = self._ffn_out(self._situ(y), x, mod[:, GATE_FFN], seg, 1.0)
        return self._post(out)
