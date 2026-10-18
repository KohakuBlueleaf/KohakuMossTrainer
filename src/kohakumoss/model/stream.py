"""Residual-stream config and the per-layer callables it binds (scaling, cast, clamp)."""

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor

from kohakumoss_common.calls import identity


@dataclass(frozen=True)
class BlockConfig:
    """One joint block: gate multipliers, writer-input scales, stream dtype, clamp."""

    gate_attn: float = 1.0
    gate_ffn: float = 1.0
    in_attn: float = 1.0
    in_ffn: float = 1.0
    stream_dtype: torch.dtype | None = None
    clamp: float | None = None


@dataclass(frozen=True)
class StreamConfig:
    """Run-time stream handling the model binds once. Identity by default."""

    text_scale: float = 1.0
    embed_scale: float = 1.0
    gate_attn: tuple[float, ...] = ()
    gate_ffn: tuple[float, ...] = ()
    in_attn: tuple[float, ...] = ()
    in_ffn: tuple[float, ...] = ()
    stream_dtype: torch.dtype | None = None
    clamp: float | None = None

    def block(self, i: int) -> BlockConfig:
        return BlockConfig(
            _at(self.gate_attn, i),
            _at(self.gate_ffn, i),
            _at(self.in_attn, i),
            _at(self.in_ffn, i),
            self.stream_dtype,
            self.clamp,
        )


def _at(values: tuple[float, ...], i: int) -> float:
    return values[i] if values else 1.0


def _cast(dtype: torch.dtype, x: Tensor) -> Tensor:
    return x.to(dtype)


def _scale_cast(scale: float, dtype: torch.dtype | None, x: Tensor) -> Tensor:
    acc = torch.promote_types(x.dtype, torch.float32)
    return (x.to(acc) * scale).to(dtype or x.dtype)


def caster(dtype: torch.dtype | None):
    """x -> x.to(dtype), or identity."""
    return identity if dtype is None else partial(_cast, dtype)


def scaler(scale: float, dtype: torch.dtype | None):
    """x -> x * scale (>= fp32 math) in `dtype` (default: x's), or identity / cast."""
    if scale == 1.0:
        return caster(dtype)
    return partial(_scale_cast, scale, dtype)


def _gate_scaled(base, g, a, x, gate, seg, r):
    return base(a, x, gate * g, seg, r)


def _input_gate_scaled(base, s, g, a, x, gate, seg, r):
    return base(a * s, x, gate * g, seg, r)


def scaled_writer(base, s_in: float, g: float):
    """Gated-residual writer with its input scaled by s_in and its gate by g."""
    if s_in == 1.0 and g == 1.0:
        return base
    if s_in == 1.0:
        return partial(_gate_scaled, base, g)
    return partial(_input_gate_scaled, base, s_in, g)


class ClampCounter:
    """Saturating clamp at +-limit; counts clamped values on the device."""

    def __init__(self, limit: float, device):
        self.limit = limit
        self.count = torch.zeros((), dtype=torch.int64, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        self.count += (x.abs() >= self.limit).sum()
        return x.clamp(-self.limit, self.limit)


def clamper(limit: float | None, device):
    return identity if limit is None else ClampCounter(limit, device)
