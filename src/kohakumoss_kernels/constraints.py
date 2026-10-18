"""Declarative `supports(query)` builders shared by backends."""

from collections.abc import Callable

import torch

from kohakumoss_kernels.registry import OpQuery


def requires(
    kinds: tuple[str, ...] = ("cuda",),
    min_sm: int = 0,
    max_sm: int = 10_000,
    dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16),
    extra: Callable[[OpQuery], str | None] | None = None,
) -> Callable[[OpQuery], str | None]:
    """A `supports` function gating on device kind, sm range, dtype and an extra check."""

    def supports(query: OpQuery) -> str | None:
        arch = query.arch
        if arch.kind not in kinds:
            return f"device kind {arch.kind} not in {kinds}"
        if arch.kind == "cuda" and not (min_sm <= arch.sm <= max_sm):
            return f"{arch.name} outside sm{min_sm}..sm{max_sm}"
        if query.dtype not in dtypes:
            return f"dtype {query.dtype} not in {dtypes}"
        if query.dtype == torch.bfloat16 and not arch.has_bf16:
            return f"{arch.name} has no bf16"
        return None if extra is None else extra(query)

    return supports
