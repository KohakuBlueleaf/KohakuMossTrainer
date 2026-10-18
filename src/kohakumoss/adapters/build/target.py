"""Target: one adapted layer (its path, LyCORIS module and kind record) and one trained
norm. A Target resolves its kind record, base-weight source and dW builder when built.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.adapters.bind.context import Draws
from kohakumoss.adapters.kinds import Kind, record_of
from kohakumoss.adapters.kinds.common import delta_builder

CHILD = "lyco"


def is_stored(linear: nn.Module) -> bool:
    """A linear whose base is quantized / streamed storage or runs through fp8
    compute copies (anything but a plain 16/32-bit dense parameter)."""
    return (
        getattr(linear, "base", None) is not None
        or getattr(linear, "compute", None) is not None
    )


def base_kind(linear: nn.Module) -> str:
    """A short name of a stored base for error messages."""
    base = getattr(linear, "base", None) or getattr(linear, "compute", None)
    return type(base).__name__ if base is not None else "dense"


def _override_base(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    return tensor.to(dtype)


def _dense_base(linear: nn.Module, dtype: torch.dtype) -> Tensor:
    return linear.weight.to(dtype)


@dataclass
class Target:
    """An adapted linear: `path` is its module path, `lora_name` its LyCORIS key;
    `autotuned` the AutotunedOps of its current bound forward."""

    path: str
    lora_name: str
    linear: nn.Module
    lyco: nn.Module
    base_override: Tensor | None = None
    backend: str | None = None
    axis: str = "joint"
    autotuned: list = field(default_factory=list)
    record: Kind = field(init=False)
    wd: bool = field(init=False)
    wd_on_out: bool = field(init=False)
    draws: Draws = field(init=False)
    base_fn: Callable = field(init=False)
    unit_fn: Callable = field(init=False)

    def __post_init__(self):
        self.record = record_of(self.lyco)
        self.wd = bool(getattr(self.lyco, "wd", False))
        self.wd_on_out = bool(getattr(self.lyco, "wd_on_out", True))
        self.draws = Draws(int(getattr(self.lyco, "lora_dim", 0) or 0))
        self.resolve_base()
        self.unit_fn = partial(self.record.diff_weight, self)

    def resolve_base(self) -> None:
        """Bind the base-weight source: override, the attached quantized / streamed
        storage (one transient dequantized copy per call), or the dense parameter."""
        if self.base_override is not None:
            self.base_fn = partial(_override_base, self.base_override)
        elif getattr(self.linear, "base", None) is not None:
            self.base_fn = self.linear.base.weight
        else:
            self.base_fn = partial(_dense_base, self.linear)

    @property
    def stored(self) -> bool:
        """The base is not a plain 16/32-bit dense parameter: quantized or streamed
        storage, or fp8 compute copies; such a base runs bypass only."""
        return is_stored(self.linear)

    def base_bias(self) -> Tensor | None:
        """The frozen base bias."""
        return self.linear.bias

    def base_weight(self, dtype: torch.dtype = torch.float32) -> Tensor:
        """The frozen base weight from the bound source."""
        return self.base_fn(dtype)

    def delta_at(self, multiplier: float) -> Callable[[], tuple[Tensor, Tensor | None]]:
        """(dW, db) builder at a fixed multiplier (fp32, adapter device)."""
        return delta_builder(self, multiplier)

    def delta(self, multiplier: float = 1.0) -> tuple[Tensor, Tensor | None]:
        """(dW, db) at `multiplier`, evaluated once (export / merge)."""
        return self.delta_at(multiplier)()


@dataclass
class NormTarget:
    """A norm whose weight (and bias) carry a trained delta (adapters.train.norms.NormDelta)."""

    path: str
    lora_name: str
    module: nn.Module
    weight_delta: nn.Module
    bias_delta: nn.Module | None = None

    def deltas(self) -> list[tuple[str, nn.Module]]:
        out = [("w_norm", self.weight_delta)]
        if self.bias_delta is not None:
            out.append(("b_norm", self.bias_delta))
        return out

    def delta(self) -> tuple[Tensor, Tensor | None]:
        w = self.weight_delta.w_norm.detach().float().cpu()
        if self.bias_delta is None:
            return w, None
        return w, self.bias_delta.w_norm.detach().float().cpu()
