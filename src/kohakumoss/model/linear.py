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
        self.compute = None
        self._impl = MossLinear._dense if bias else MossLinear._dense_nobias

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

    def set_compute(self, cache) -> None:
        """Run the dense (trainable) weight through fp8 copies held by `cache`
        (`kohakumoss.quant.compute.ComputeCache`), or None for the 16-bit forward."""
        self.compute = cache
        self.rebind()

    def rebind(self) -> None:
        """Re-select the forward after a structural change (bias, base, fp8 compute):
        `_impl` is the class function, called as `self._impl(self, x)`."""
        match (self.base is not None, self.compute is not None, self.bias is not None):
            case (True, _, True):
                self._impl = MossLinear._stored
            case (True, _, False):
                self._impl = MossLinear._stored_nobias
            case (False, True, True):
                self._impl = MossLinear._fp8
            case (False, True, False):
                self._impl = MossLinear._fp8_nobias
            case (False, False, True):
                self._impl = MossLinear._dense
            case (False, False, False):
                self._impl = MossLinear._dense_nobias

    def _dense(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))

    def _dense_nobias(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype))

    def _stored(self, x: Tensor) -> Tensor:
        return self.base.linear(x, self.bias.to(x.dtype))

    def _stored_nobias(self, x: Tensor) -> Tensor:
        return self.base.linear(x, None)

    def _fp8(self, x: Tensor) -> Tensor:
        return self.compute.linear(x, self.weight, self.bias)

    def _fp8_nobias(self, x: Tensor) -> Tensor:
        return self.compute.linear(x, self.weight, None)

    def forward(self, x: Tensor) -> Tensor:
        return self._impl(self, x)

    def __delattr__(self, name: str) -> None:
        if name == "weight" and "weight" in self._parameters:
            w = self._parameters["weight"]
            self.__dict__["_held"] = torch.empty(0, dtype=w.dtype, device=w.device)
        super().__delattr__(name)

    def _apply(self, fn, recurse: bool = True):
        held = self.__dict__.get("_held")
        if held is not None:
            self.__dict__["_held"] = fn(held)
        return super()._apply(fn, recurse)

    @property
    def weight_like(self) -> Tensor:
        """The weight, or an empty tensor of its dtype / device after the weight
        parameter was deleted from this module."""
        weight = self._parameters.get("weight")
        return weight if weight is not None else self.__dict__["_held"]

    @property
    def is_plain(self) -> bool:
        """Dense storage, 16-bit compute and no adapter (or other) forward override."""
        return (
            self.base is None and self.compute is None and "forward" not in vars(self)
        )
