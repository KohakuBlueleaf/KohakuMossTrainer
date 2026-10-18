"""Parameter dtypes: the trunk in the compute dtype, listed modules kept in fp32."""

import torch
from torch import nn


def _kept(name: str, keep_fp32: tuple[str, ...]) -> bool:
    return any(name == k or name.startswith(k + ".") for k in keep_fp32)


def cast_params(
    model: nn.Module, dtype: torch.dtype, keep_fp32: tuple[str, ...] = ()
) -> None:
    """Cast each module's own tensors once: fp32 for listed paths (and their children),
    `dtype` elsewhere. Kept modules never pass through `dtype`."""
    for name, module in model.named_modules():
        target = torch.float32 if _kept(name, keep_fp32) else dtype
        module._apply(
            lambda t, d=target: t.to(d) if t.is_floating_point() else t, False
        )
