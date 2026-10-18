"""Streaming safetensors writer: the header from shapes / dtypes, then one tensor at a
time copied to the host and written. Peak extra host memory is one tensor. See
docs/internals/checkpoints.md."""

import json
from pathlib import Path

import torch
from torch import Tensor

DTYPES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
}


def header(state: dict[str, Tensor], metadata: dict[str, str] | None = None) -> bytes:
    """The safetensors prefix: 8-byte little-endian length, then the JSON header
    (padded to 8 bytes) with the tensors laid out in `state` order."""
    head: dict = {"__metadata__": dict(metadata)} if metadata else {}
    offset = 0
    for name, t in state.items():
        n = t.nbytes
        head[name] = {
            "dtype": DTYPES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + n],
        }
        offset += n
    raw = json.dumps(head, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return len(raw).to_bytes(8, "little") + raw


def host_bytes(t: Tensor):
    """`t`'s bytes as a host uint8 array (a copy only for device / strided tensors)."""
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()


def save_stream(
    state: dict[str, Tensor], path: str | Path, metadata: dict[str, str] | None = None
) -> None:
    """Write `state` as one safetensors file, tensor by tensor."""
    with open(path, "wb") as f:
        f.write(header(state, metadata))
        for t in state.values():
            if t.numel():
                f.write(host_bytes(t))
