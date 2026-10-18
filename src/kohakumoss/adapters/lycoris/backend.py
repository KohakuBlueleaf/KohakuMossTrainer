"""LyCORIS kernel backend for one device, chosen once per apply. See
docs/concepts/adapters.md."""

import torch
from lycoris.kernels.dispatch import available_backends, resolve_backend


def pick_backend(device: torch.device | str, requested: str = "auto") -> str:
    """`triton` | `tilelang` | `compile` | `torch` for adapters on `device`.

    `auto` takes LyCORIS's per-arch order on CUDA and the eager bodies elsewhere; a
    named backend the device cannot run raises ValueError.
    """
    device = torch.device(device)
    if device.type != "cuda":
        if requested in ("auto", "torch"):
            return "torch"
        if requested == "compile":
            return requested
        raise ValueError(
            f"KERNELS['lycoris']={requested!r} needs a CUDA device; adapters are on "
            f"{device} (use 'auto', 'torch' or 'compile')"
        )
    if requested != "auto" and requested not in available_backends(device):
        raise ValueError(
            f"KERNELS['lycoris']={requested!r} is not available on {device}; "
            f"available: {available_backends(device)}"
        )
    return resolve_backend(None if requested == "auto" else requested, device)
