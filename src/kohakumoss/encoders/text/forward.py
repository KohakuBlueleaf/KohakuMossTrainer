"""Qwen3.5 text-model forward over packed prompts on kohakumoss_kernels ops.

Prompts are concatenated on one token axis with `cu_seqlens` boundaries and positions
reset per prompt; the Gated DeltaNet conv / delta rule / gated norm and the causal GQA
attention read the boundaries, every other op is per token. Layers run layer-major
over all packs of a pass, so a streamed layer crosses the bus once per pass. Each
decoder layer's block function is bound once from its layer type. See
docs/internals/text-encoder.md.
"""

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.logging.log import get_logger
from kohakumoss_common.calls import nothing
from kohakumoss_kernels import registry
from kohakumoss_kernels.errors import describe
from kohakumoss_kernels.hardware.current import rng_kept
from kohakumoss_kernels.hardware.memory import release

log = get_logger(__name__)

TEXT_OPS = (
    "rms_norm",
    "causal_conv_varlen",
    "gated_delta_rule_varlen",
    "rms_norm_gated",
    "attention_varlen_causal",
)


@dataclass
class Pack:
    """One kernel call's prompts: `ids` int64 [N] (host), `cu` int32 [B+1] and `pos`
    int64 [N] on the device, the longest row, and `take` (padded rows: the real-token
    indices; None when every token is real)."""

    ids: Tensor
    cu: Tensor
    pos: Tensor
    max_len: int
    lengths: list[int]
    take: Tensor | None = None


def make_pack(rows: list[list[int]], device, pad_id: int, padded: bool) -> Pack:
    """Pack token rows: concatenated (`padded=False`) or right-padded to the longest."""
    lengths = [len(r) for r in rows]
    width = max(lengths)
    if padded:
        flat = [t for r in rows for t in r + [pad_id] * (width - len(r))]
        seg = [width] * len(rows)
        take = torch.cat(
            [torch.arange(n) + i * width for i, n in enumerate(lengths)]
        ).to(device)
    else:
        flat = [t for r in rows for t in r]
        seg = lengths
        take = None
    cu = torch.tensor([0] + seg, dtype=torch.int64).cumsum(0).to(torch.int32)
    pos = torch.cat([torch.arange(n) for n in seg])
    return Pack(
        torch.tensor(flat, dtype=torch.int64),
        cu.to(device),
        pos.to(device),
        width,
        lengths,
        take,
    )


def op_static(config) -> dict[str, dict]:
    """Static query options (head dims and counts) of each text op for a Qwen3.5 text
    config."""
    return {
        "gated_delta_rule_varlen": {
            "head_dim_k": config.linear_key_head_dim,
            "head_dim_v": config.linear_value_head_dim,
            "heads_k": config.linear_num_key_heads,
            "heads_v": config.linear_num_value_heads,
        },
        "attention_varlen_causal": {
            "head_dim": config.head_dim,
            "heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
        },
    }


def probe_args(name: str, device, dtype: torch.dtype, static: dict) -> tuple:
    """Two-segment inputs of op `name` at its static head dims (the build-time backend
    probe)."""
    cu = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)

    def r(*shape):
        return torch.randn(*shape, device=device).to(dtype)

    match name:
        case "rms_norm":
            return r(8, 64), torch.ones(64, device=device), 1e-6
        case "causal_conv_varlen":
            return r(8, 64), r(64, 4), cu
        case "rms_norm_gated":
            return r(8, 64), r(8, 64), r(64), 1e-6
        case "gated_delta_rule_varlen":
            dk, dv = static.get("head_dim_k", 128), static.get("head_dim_v", 128)
            hk, hv = static.get("heads_k", 2), static.get("heads_v", 4)
            return (
                r(8, hk, dk), r(8, hk, dk), r(8, hv, dv), r(8, hv), r(8, hv),
                torch.zeros(hv, device=device), torch.zeros(hv, device=device), cu, 5,
            )  # fmt: skip
        case _:
            d = static.get("head_dim", 128)
            h, kv = static.get("heads", 4), static.get("kv_heads", 2)
            return r(8, h, d), r(8, kv, d), r(8, kv, d), cu, 5, d**-0.5


def probe(fn, name: str, device, dtype: torch.dtype, opts: dict) -> None:
    """Run `fn` once on the probe inputs of op `name`; an out-of-memory retries once
    after the device's free blocks go back to the driver, then propagates."""
    for retry in (False, True):
        try:
            with torch.no_grad(), rng_kept([device]):
                fn(*probe_args(name, device, dtype, opts))
            return
        except torch.OutOfMemoryError:
            if retry:
                raise
            release(device)


class TextOps:
    """The text-encoder ops resolved once for (device, dtype), with backend overrides
    (`kernels`: op -> backend) and per-op static options. An accelerator op whose
    build-time probe (`probe`: an out-of-memory retried once after a cache release)
    raises a RuntimeError (out-of-memory included) binds the eager reference, logged at
    WARNING and kept in `fallbacks` (never written to disk); any other error
    propagates. See docs/internals/kernel-dev.md#fallback."""

    def __init__(
        self,
        device,
        dtype: torch.dtype,
        kernels: dict | None = None,
        static: dict | None = None,
    ):
        kernels, static = dict(kernels or {}), dict(static or {})
        device = torch.device(device)
        self.names, self.fallbacks = {}, {}
        for name in TEXT_OPS:
            opts = static.get(name, {})
            fn = registry.resolve(name, device, dtype, kernels.get(name), **opts)
            if device.type != "cpu" and name not in kernels:
                try:
                    probe(fn, name, device, dtype, opts)
                except RuntimeError as err:
                    why = self.fallbacks[name] = describe(err)[:300]
                    log.warning(
                        f"text encoder op {name} on {device}: backend {fn.backend} "
                        f"failed its probe, using eager: {why}"
                    )
                    fn = registry.resolve(name, device, dtype, "eager", **opts)
            setattr(self, name, fn)
            self.names[name] = getattr(fn, "__name__", repr(fn))

    def norm(self, x: Tensor, mod: nn.Module) -> Tensor:
        """Qwen3.5 zero-centred RMSNorm: x / rms(x) * (1 + w), fp32 math."""
        return self.rms_norm(x, 1.0 + mod.weight.float(), mod.eps)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    return torch.cat((xr * cos + _rotate_half(xr) * sin, xp), dim=-1)


def attention_block(layer, ops: TextOps, h: Tensor, pk: Pack, rope) -> Tensor:
    att = layer.self_attn
    n, hd = h.shape[0], att.head_dim
    x = ops.norm(h, layer.input_layernorm)
    qg = att.q_proj(x).view(n, -1, 2 * hd)
    gate = qg[..., hd:].reshape(n, -1)
    q = ops.norm(qg[..., :hd], att.q_norm)
    k = ops.norm(att.k_proj(x).view(n, -1, hd), att.k_norm)
    v = att.v_proj(x).view(n, -1, hd)
    cos, sin = rope
    q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
    o = ops.attention_varlen_causal(q, k, v, pk.cu, pk.max_len, att.scaling)
    h = h + att.o_proj(o.reshape(n, -1) * torch.sigmoid(gate))
    return h + layer.mlp(ops.norm(h, layer.post_attention_layernorm))


def delta_block(layer, ops: TextOps, h: Tensor, pk: Pack, rope) -> Tensor:
    la = layer.linear_attn
    n = h.shape[0]
    x = ops.norm(h, layer.input_layernorm)
    qkv = ops.causal_conv_varlen(la.in_proj_qkv(x), la.conv1d.weight[:, 0], pk.cu)
    q, k, v = qkv.split([la.key_dim, la.key_dim, la.value_dim], dim=-1)
    o = ops.gated_delta_rule_varlen(
        q.view(n, -1, la.head_k_dim),
        k.view(n, -1, la.head_k_dim),
        v.view(n, -1, la.head_v_dim),
        la.in_proj_a(x),
        la.in_proj_b(x),
        la.A_log,
        la.dt_bias,
        pk.cu,
        pk.max_len,
    )
    z = la.in_proj_z(x)
    o = ops.rms_norm_gated(
        o.reshape(-1, la.head_v_dim),
        z.reshape(-1, la.head_v_dim),
        la.norm.weight,
        la.norm.variance_epsilon,
    )
    h = h + la.out_proj(o.reshape(n, -1))
    return h + layer.mlp(ops.norm(h, layer.post_attention_layernorm))


BLOCKS = {"full_attention": attention_block, "linear_attention": delta_block}


class KernelForward:
    """Final hidden states [N, D] of every pack (real tokens only), layer-major."""

    def __init__(self, text_model: nn.Module, placement, ops: TextOps):
        self.text = text_model
        self.placement = placement
        types = text_model.config.layer_types
        self.blocks = [
            partial(BLOCKS[types[i]], layer, ops)
            for i, layer in enumerate(text_model.layers)
        ]
        self.ops = ops

    unit = "layer"

    def units(self, packs: list[Pack]) -> int:
        """Progress units of one call: its layers."""
        return len(self.blocks)

    def __call__(self, packs: list[Pack], tick=nothing) -> list[Tensor]:
        """`tick()` after each layer (over every pack)."""
        place = self.placement
        hs = [place.embed(p.ids) for p in packs]
        ropes = [self.text.rotary_emb(h, p.pos[None]) for h, p in zip(hs, packs)]
        ropes = [(c[0][:, None, :], s[0][:, None, :]) for c, s in ropes]
        for i, block in enumerate(self.blocks):
            place.enter[i]()
            hs = [block(h, p, r) for h, p, r in zip(hs, packs, ropes)]
            place.leave[i]()
            tick()
        out = []
        for h, p in zip(hs, packs):
            h = self.ops.norm(h, self.text.norm)
            out.append(h if p.take is None else h.index_select(0, p.take))
        return out
