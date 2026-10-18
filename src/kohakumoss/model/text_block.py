"""Text refiner: projection from the text encoder width plus unmodulated blocks."""

from functools import partial

from torch import Tensor, nn

from kohakumoss.model.components.ffn import situ_pair
from kohakumoss.model.components.norm import RMSNorm
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.opset import OpSet
from kohakumoss.model.stream import StreamConfig, caster, identity, scaler
from kohakumoss.types.arch import MossArchConfig
from kohakumoss.types.seqinfo import SeqInfo


class TextBlock(nn.Module):
    """Pre-norm self-attention (biased fused qkv, qk-norm, RoPE) + SiTU FFN, no gates."""

    def __init__(self, cfg: MossArchConfig, device=None, dtype=None):
        super().__init__()
        dim, ffn = cfg.dim, cfg.ffn_dim
        kw = dict(device=device, dtype=dtype)
        self.heads = cfg.heads
        self.norm1 = RMSNorm(dim, cfg.eps, **kw)
        self.norm2 = RMSNorm(dim, cfg.eps, **kw)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.eps, **kw)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.eps, **kw)
        self.qkv = MossLinear(dim, 3 * dim, **kw)
        self.proj = MossLinear(dim, dim, **kw)
        self.ffn_gate = MossLinear(dim, ffn, **kw)
        self.ffn_up = MossLinear(dim, ffn, **kw)
        self.ffn_down = MossLinear(ffn, dim, **kw)

    def bind(self, ops: OpSet, stream: StreamConfig = StreamConfig()) -> None:
        """Writers scaled by the stream's text scale; norm outputs cast for an fp32
        stream."""
        self.ops = ops
        c = stream.text_scale
        self._attn_out = _text_writer(ops, self.proj, c)
        self._ffn_out = _text_writer(ops, self.ffn_down, c)
        compute = self.qkv.weight_like.dtype
        self._cast = caster(compute if stream.stream_dtype is not None else None)

    def forward(self, x: Tensor, seq: SeqInfo, bands: Tensor) -> Tensor:
        ops = self.ops
        y = self._cast(ops.rms_norm(x, self.norm1.weight, self.norm1.eps))
        q, k, v = (t.unflatten(-1, (self.heads, -1)) for t in self.qkv(y).chunk(3, -1))
        q, k = ops.qk_norm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            seq.pos_txt,
            bands,
            self.q_norm.eps,
        )
        o = ops.attention(q, k, v, seq.attn_text).flatten(1)
        x = self._attn_out(o, x, None, None, 1.0)
        y = self._cast(ops.rms_norm(x, self.norm2.weight, self.norm2.eps))
        a = situ_pair(self.ffn_up(y), self.ffn_gate(y))
        return self._ffn_out(a, x, None, None, 1.0)


def _scaled_text_write(op, lin, c, a, x, gate, seg, r):
    return op(x, lin(a) * c, gate, seg, r)


def _text_writer(ops: OpSet, lin: MossLinear, c: float):
    if c == 1.0:
        return ops.bind_gated_residual(lin)
    return partial(_scaled_text_write, ops.gated_residual, lin, c)


class TextEmbedder(nn.Module):
    """`proj` (text_dim -> dim, biased) followed by `blocks` over packed text tokens."""

    def __init__(self, cfg: MossArchConfig, device=None, dtype=None):
        super().__init__()
        self.proj = MossLinear(cfg.text_dim, cfg.dim, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            TextBlock(cfg, device=device, dtype=dtype) for _ in range(cfg.text_blocks)
        )
        self._in = identity

    def bind(self, ops: OpSet, stream: StreamConfig = StreamConfig()) -> None:
        self._in = scaler(stream.text_scale, stream.stream_dtype)
        for block in self.blocks:
            block.bind(ops, stream)

    def forward(self, text: Tensor, seq: SeqInfo, bands: Tensor) -> Tensor:
        x = self._in(self.proj(text.to(self.proj.weight_like.dtype)))
        for block in self.blocks:
            x = block(x, seq, bands)
        return x
