"""Forward ops whose outputs a selective activation checkpoint keeps: the attention
forwards (kernel and torch varlen) and the fused GEMM + epilogue forwards, each added by
its backend at import (`keep`); `on_dispatch` / `require_dispatch` switch ops with an
eager direct path back to their custom ops while a selective checkpoint needs to see
them. See docs/internals/kernel-dev.md#selective-checkpoint.
"""

from collections.abc import Callable

import torch

KEPT: set = set()
SWITCHES: list[Callable[[bool], None]] = []


def on_dispatch(switch: Callable[[bool], None]) -> None:
    """Register `switch(dispatch)`, called by `require_dispatch`."""
    SWITCHES.append(switch)


def require_dispatch(dispatch: bool = True) -> None:
    """Route every op with an eager direct path through its custom op (`dispatch`), or
    back to the direct path."""
    for switch in SWITCHES:
        switch(dispatch)


def keep(*ops) -> None:
    """Add op overloads (e.g. `torch.ops.kohakumoss.<name>.default`) to `KEPT`."""
    KEPT.update(ops)


def keep_named(namespace: str, *names: str) -> None:
    """Add `torch.ops.<namespace>.<name>.default` for each registered `name`."""
    space = getattr(torch.ops, namespace)
    keep(*(getattr(space, name).default for name in names if hasattr(space, name)))
