"""Kind: everything the applier does differently per adapter algorithm, as one record.

A record is looked up once, when its Target is built; every later decision reads the
record's callables. `register` adds an algorithm without touching the applier.
"""

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

STACKING = ("additive", "commuting", "chained", "dense")


def no_path(*args):
    return None


def always(m: float) -> bool:
    return True


@dataclass(frozen=True)
class Kind:
    """The per-algorithm behaviour of one LyCORIS module class.

    `algos` maps each ALGO name to the LyCORIS network_module that builds it. Every
    callable is documented on its field in docs/internals/adapter-kinds.md.
    """

    name: str
    module: type
    algos: dict[str, str]
    unit: Callable
    at: Callable | None = None
    fused_unit: Callable = no_path
    bypass: Callable = no_path
    merge: Callable | None = None
    eager: Callable | None = None
    special: Callable | None = None
    comfy: Callable | None = None
    row_names: Callable = no_path
    stacking: str = "dense"
    exact_at: Callable[[float], bool] = always
    up_side: tuple[str, ...] = ()
    token_axis: bool = False
    dora: bool = False
    training_dependent: bool = False
    draw: Callable | None = None
    from_comfy: Callable = no_path
    owns_weight: bool = False
    dimless: bool = False


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
