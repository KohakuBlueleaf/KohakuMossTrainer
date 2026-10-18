"""Forward ops whose outputs a selective activation checkpoint keeps: the attention
forwards (kernel and torch varlen) and the fused GEMM + epilogue forwards, each added by
its backend at import (`keep`). See docs/internals/kernel-dev.md#selective-checkpoint.
"""

import torch

KEPT: set = set()


def keep(*ops) -> None:
    """Add op overloads (e.g. `torch.ops.kohakumoss.<name>.default`) to `KEPT`."""
    KEPT.update(ops)


def keep_named(namespace: str, *names: str) -> None:
    """Add `torch.ops.<namespace>.<name>.default` for each registered `name`."""
    space = getattr(torch.ops, namespace)
    keep(*(getattr(space, name).default for name in names if hasattr(space, name)))
