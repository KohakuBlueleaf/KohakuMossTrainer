"""Kind: everything the applier does differently per adapter algorithm, as one record.

A record is looked up once, when its Target is built; every later decision reads the
record's callables. `register` adds an algorithm without touching the applier.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from torch import nn

from kohakumoss_common.calls import nothing


class Stacking(str, Enum):
    """How a kind's delta composes with the base and with other stacked adapters (see
    docs/internals/adapter-kinds.md)."""

    ADDITIVE = "additive"
    COMMUTING = "commuting"
    CHAINED = "chained"
    DENSE = "dense"


STACKING = tuple(s.value for s in Stacking)


def always(m: float) -> bool:
    return True


@dataclass(frozen=True)
class Kind:
    """The per-algorithm behaviour of one LyCORIS module class (`t` a Target, `ctx` a
    BindCtx, `ops` its kernel ops; see docs/internals/adapter-kinds.md)."""

    name: str
    module: type
    # ALGO name -> the LyCORIS network_module that builds it.
    algos: dict[str, str]
    # t -> dW at multiplier 1 (fp32, the layer's shape).
    diff_weight: Callable
    # (t, m) -> a (dW, db) builder at multiplier m, or None where m * dW is exact.
    diff_weight_at: Callable | None = None
    # (t, ops) -> W -> dW at multiplier 1 from the kernel ops, or None.
    kernel_diff_weight: Callable = nothing
    # (t, ops) -> W -> W + dW (the base add inside the op), or None.
    kernel_merged_weight: Callable = nothing
    # (t, ctx) -> the bypass forward from the kernel ops, or None.
    kernel_bypass: Callable = nothing
    # (t, ctx) -> the merge forward, in place of the default W + dW one.
    kernel_merge: Callable | None = None
    # (t, ctx) -> the forward over module methods, in place of the module bypass.
    module_forward: Callable | None = None
    # (t, ctx) -> the whole forward, in place of every other path.
    own_forward: Callable | None = None
    # t -> the tensors ComfyUI evaluates to dW, in place of the native set.
    comfy_tensors: Callable | None = None
    # (lyco, file tensors) -> undo our comfy export's conversions on a loaded module.
    undo_comfy: Callable = nothing
    # tensors -> the names whose uniform scaling scales dW.
    uniform_scale_tensors: Callable = nothing
    # tensors -> the names whose output rows scale dW's rows.
    row_scale_tensors: Callable = nothing
    stacking: Stacking = Stacking.DENSE
    # m -> whether the bypass at multiplier m equals the merged delta at m.
    bypass_exact_at: Callable[[float], bool] = always
    # The zero-initialised factors the LoRA+ ratio applies to.
    loraplus_tensors: tuple[str, ...] = ()
    # Ranks masked per token (T-LoRA).
    per_token_mask: bool = False
    supports_dora: bool = False
    training_dependent: bool = False
    # (t, generator) -> this step's random draws of the layer.
    draw_step: Callable | None = None
    # Rankless kind: DIM is ignored.
    rankless: bool = False


REGISTRY: dict[str, Kind] = {}
BY_MODULE: dict[type, Kind] = {}
BY_ALGO: dict[str, Kind] = {}


def register(kind: Kind) -> Kind:
    """Add (or replace) an algorithm record."""
    if kind.stacking not in STACKING:
        raise ValueError(f"{kind.name}: stacking {kind.stacking!r} not in {STACKING}")
    REGISTRY[kind.name] = kind
    BY_MODULE[kind.module] = kind
    for algo in kind.algos:
        BY_ALGO[algo] = kind
    return kind


def record_of(lyco: nn.Module) -> Kind:
    """The record of a LyCORIS module instance."""
    return BY_MODULE[type(lyco)]


def lycoris_algo(algo: str) -> str:
    """The LyCORIS network_module name of an ALGO."""
    return BY_ALGO[algo].algos[algo]
