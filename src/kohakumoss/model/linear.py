"""MossLinear: an nn.Linear whose base-weight storage is swappable.

It subclasses nn.Linear so LyCORIS targets it like any Linear; state-dict names stay
`weight` / `bias` so HF and ComfyUI keys load unchanged. Weights are initialised by the
model builder (`init_weights`), not at construction.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.types.protocols import BaseWeight


class MossLinear(nn.Linear):
    """y = x W^T + b with W from the dense parameter or the attached base storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.base: BaseWeight | None = None
        self._impl = self._dense

    def reset_parameters(self) -> None:
        """No-op: the model builder initialises or loads weights."""

    def init_weights(self, std: float = 0.02) -> None:
        nn.init.normal_(self.weight, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def base_weight(self, dtype: torch.dtype) -> Tensor:
        """The base weight in `dtype`, from the dense parameter or the attached storage."""
        if self.base is not None:
            return self.base.weight(dtype)
        return self.weight.to(dtype)

    def set_base(self, base: BaseWeight | None) -> None:
        """Attach quantized/streamed storage (or None for the dense parameter)."""
        self.base = base
        self._impl = self._dense if base is None else self._stored

    def _dense(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), bias)

    def _stored(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.base.weight(x.dtype), bias)

    def forward(self, x: Tensor) -> Tensor:
        return self._impl(x)

    @property
    def is_plain(self) -> bool:
        """Dense storage and no LyCORIS (or other) forward override attached."""
        return self.base is None and "forward" not in vars(self)
