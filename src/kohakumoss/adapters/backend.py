"""LyCORIS kernel tier for one device, chosen once per apply. See docs/concepts/adapters.md."""

import torch
from lycoris.kernels.dispatch import resolve_backend

FUSED = ("triton", "tilelang")


def pick_backend(device: torch.device | str, requested: str = "auto") -> str:
    """`triton` | `tilelang` | `compile` | `torch` for adapters on `device`.

    The per-arch order is LyCORIS's (tilelang first on sm70/75, triton from sm80);
    non-CUDA devices take the eager bodies.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return "torch" if requested == "auto" else requested
    return resolve_backend(None if requested == "auto" else requested, device)


def is_fused(backend: str) -> bool:
    return backend in FUSED
