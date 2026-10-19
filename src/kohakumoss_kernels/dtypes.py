"""Names of the torch dtypes the kernels take: knob names (`BY_NAME` / `NAME_OF`),
safetensors header names (`SAFETENSORS`) and torch's own name without the module prefix
(`short`), as problem keys spell it."""

import torch

BY_NAME: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
NAME_OF: dict[torch.dtype, str] = {dtype: name for name, dtype in BY_NAME.items()}
SAFETENSORS: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


def short(dtype: torch.dtype) -> str:
    """`float16` for torch.float16: torch's dtype name without `torch.`."""
    return str(dtype).removeprefix("torch.")
