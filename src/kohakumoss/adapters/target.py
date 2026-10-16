"""Target: one adapted layer (its path, LyCORIS module and kind) and one trained norm.

A Target's base-weight source is resolved once (`base_fn`): an override tensor, the
cached original of `full`, or the linear's own `base_weight`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
from lycoris.modules import (
    ButterflyOFTModule,
    DiagOFTModule,
    DyLoraModule,
    FullModule,
    GLoRAModule,
    IA3Module,
    LoConModule,
    LohaModule,
    LokrModule,
    TLoraModule,
)
from torch import Tensor, nn

KINDS = {
    LoConModule: "locon",
    LohaModule: "loha",
    LokrModule: "lokr",
    DiagOFTModule: "oft",
    ButterflyOFTModule: "boft",
    GLoRAModule: "glora",
    IA3Module: "ia3",
    DyLoraModule: "dylora",
    TLoraModule: "tlora",
    FullModule: "full",
}
CHILD = "lyco"


def kind_of(lyco: nn.Module) -> str:
    """Algorithm kind of a LyCORIS module instance."""
    return KINDS[type(lyco)]


def _override_base(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    return tensor.to(dtype)


def _full_base(lyco: nn.Module, dtype: torch.dtype) -> Tensor:
    return lyco.org_weight.to(device=lyco.weight.device, dtype=dtype)


@dataclass
class Target:
    """An adapted linear: `path` is its module path, `lora_name` its LyCORIS key."""

    path: str
    lora_name: str
    linear: nn.Module
    lyco: nn.Module
    kind: str
    wd: bool = False
    wd_on_out: bool = True
    fused: bool = False
    base_override: Tensor | None = None
    base_fn: Callable | None = None
    unit_fn: Callable | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.resolve_base()

    def resolve_base(self) -> None:
        """Bind the base-weight source; call again after changing `base_override`."""
        if self.base_override is not None:
            self.base_fn = partial(_override_base, self.base_override)
        elif self.kind == "full":
            self.base_fn = partial(_full_base, self.lyco)
        else:
            self.base_fn = self.linear.base_weight

    def set_override(self, tensor: Tensor | None) -> None:
        self.base_override = tensor
        self.resolve_base()

    @property
    def stored(self) -> bool:
        """The base weight lives in quantized / streamed storage, not a dense parameter."""
        return getattr(self.linear, "base", None) is not None

    def base_bias(self) -> Tensor | None:
        """The frozen base bias (the cached original for `full`)."""
        if self.kind == "full":
            org = self.lyco.org_bias
            return None if not org else org[0]
        return self.linear.bias

    def base_weight(self, dtype: torch.dtype = torch.float32) -> Tensor:
        """The frozen base weight from the bound source."""
        return self.base_fn(dtype)


@dataclass
class NormTarget:
    """A norm whose weight (and bias) carry a trained delta (adapters.norms.NormDelta)."""

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
