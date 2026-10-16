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
        self._impl = self._dense if bias else self._dense_nobias

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
        """Attach quantized/streamed storage, or None to go back to the dense parameter."""
        self.clear_base()
        if base is not None:
            self.base = base
            self.rebind()

    def clear_base(self) -> None:
        """Detach any base storage (Module-typed or not) and use the dense parameter."""
        self._modules.pop("base", None)
        self.__dict__["base"] = None
        self.rebind()

    def rebind(self) -> None:
        """Re-select the forward after a structural change (bias added/removed, base)."""
        match (self.base is not None, self.bias is not None):
            case (True, True):
                self._impl = self._stored
            case (True, False):
                self._impl = self._stored_nobias
            case (False, True):
                self._impl = self._dense
            case (False, False):
                self._impl = self._dense_nobias

    def _dense(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))

    def _dense_nobias(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype))

    def _stored(self, x: Tensor) -> Tensor:
        return self.base.linear(x, self.bias.to(x.dtype))

    def _stored_nobias(self, x: Tensor) -> Tensor:
        return self.base.linear(x, None)

    def forward(self, x: Tensor) -> Tensor:
        return self._impl(x)

    @property
    def is_plain(self) -> bool:
        """Dense storage and no LyCORIS (or other) forward override attached."""
        return self.base is None and "forward" not in vars(self)
