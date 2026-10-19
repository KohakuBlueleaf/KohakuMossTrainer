"""Target: one adapted layer (its path, LyCORIS module and kind record) and one trained
norm. A Target resolves its kind record, base-weight source and dW builder when built.
The layer is held weakly: it holds the target's module and bound forward. See
docs/internals/reference-cycles.md#adapters.
"""

import types
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.adapters.bind.context import Draws
from kohakumoss.adapters.kinds import Kind, record_of
from kohakumoss.adapters.kinds.common import delta_builder
from kohakumoss_common.weak import weak_method

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


def base_forward(layer: weakref.ref, x: Tensor) -> Tensor:
    """The class forward of the layer `layer` refers to (never an adapter forward in
    its instance dictionary) on `x`."""
    module = layer()
    return type(module).forward(module, x)


class WeakLayer:
    """LyCORIS's `org_module` (a one-item list of the layer) over a weak reference."""

    __slots__ = ("ref",)

    def __init__(self, ref: weakref.ref):
        self.ref = ref

    def __getitem__(self, index: int) -> nn.Module:
        if index not in (0, -1):
            raise IndexError(index)
        return self.ref()

    def __len__(self) -> int:
        return 1


HOOK_DICTS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


def _weakened(hook, module: nn.Module):
    """`hook` over a weak reference when it is a bound method of `module` (bare, or
    inside torch's `_WrappedHook`), else None."""
    inner = getattr(hook, "hook", hook)
    if not isinstance(inner, types.MethodType) or inner.__self__ is not module:
        return None
    if inner is hook:
        return weak_method(hook)
    hook.hook = hook.__wrapped__ = weak_method(inner)
    return hook


def weak_self_hooks(root: nn.Module) -> None:
    """Every hook of `root` and its submodules that is a bound method of its own module
    (LyCORIS registers its state-dict hooks so) is held over a weak reference."""
    for module in root.modules():
        for name in HOOK_DICTS:
            hooks = getattr(module, name, None)
            for key, hook in list((hooks or {}).items()):
                weak = _weakened(hook, module)
                if weak is not None:
                    hooks[key] = weak


def weak_org(lyco: nn.Module, linear: nn.Module) -> None:
    """Point the LyCORIS module's `org_module` / `org_forward` at `linear` weakly and
    hold its own bound-method hooks weakly."""
    ref = weakref.ref(linear)
    lyco.org_module = WeakLayer(ref)
    lyco.org_forward = partial(base_forward, ref)
    weak_self_hooks(lyco)


def _override_base(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    return tensor.to(dtype)


def _dense_base(layer: weakref.ref, dtype: torch.dtype) -> Tensor:
    return layer().weight.to(dtype)


class Target:
    """An adapted linear: `path` is its module path, `lora_name` its LyCORIS key;
    `layer` a weak reference to the linear (`linear` dereferences it); `autotuned` the
    AutotunedOps of its current bound forward."""

    def __init__(
        self,
        path: str,
        lora_name: str,
        linear: nn.Module,
        lyco: nn.Module,
        base_override: Tensor | None = None,
        backend: str | None = None,
        axis: str = "joint",
    ):
        self.path = path
        self.lora_name = lora_name
        self.layer = weakref.ref(linear)
        self.lyco = lyco
        self.base_override = base_override
        self.backend = backend
        self.axis = axis
        self.autotuned: list = []
        self.record: Kind = record_of(lyco)
        self.wd = bool(getattr(lyco, "wd", False))
        self.wd_on_out = bool(getattr(lyco, "wd_on_out", True))
        self.draws = Draws(int(getattr(lyco, "lora_dim", 0) or 0))
        self.resolve_base()

    @property
    def linear(self) -> nn.Module:
        """The adapted layer."""
        return self.layer()

    def resolve_base(self) -> None:
        """Bind the base-weight source: override, the attached quantized / streamed
        storage (one transient dequantized copy per call), or the dense parameter."""
        linear = self.linear
        if self.base_override is not None:
            self.base_fn = partial(_override_base, self.base_override)
        elif getattr(linear, "base", None) is not None:
            self.base_fn = linear.base.weight
        else:
            self.base_fn = partial(_dense_base, self.layer)

    def unit_fn(self) -> Tensor:
        """dW at multiplier 1 from the kind record."""
        return self.record.diff_weight(self)

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
