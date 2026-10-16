"""TensorStore: a KVault of tensors with a small binary header.

Value layout: magic `KMT1`, dtype code u8, ndim u8, pad u16, scale f32, shape u32 x ndim,
then the raw little-endian payload. fp8 values are stored as e4m3 with a per-tensor
scale (absmax / 448). Metadata key `m:<key>` holds the first dim (token count).
"""

import struct
from functools import partial

import kohakuvault as kv
import numpy as np
import torch
from torch import Tensor

MAGIC = b"KMT1"
HEAD = struct.Struct("<4sBBHf")
STORAGE = {
    "bf16": (1, torch.bfloat16),
    "fp16": (2, torch.float16),
    "fp32": (3, torch.float32),
    "fp8": (4, torch.float8_e4m3fn),
}
CODES = {code: (name, dtype) for name, (code, dtype) in STORAGE.items()}
FP8_MAX = 448.0


def _pack(code: int, scale: float, t: Tensor, payload: Tensor) -> bytes:
    raw = payload.view(torch.uint8).numpy().tobytes()
    shape = struct.pack(f"<{t.ndim}I", *t.shape)
    return HEAD.pack(MAGIC, code, t.ndim, 0, scale) + shape + raw


def _encode_plain(code: int, dtype: torch.dtype, t: Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous()
    return _pack(code, 1.0, t, t.to(dtype))


def _encode_fp8(code: int, dtype: torch.dtype, t: Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous().float()
    absmax = float(t.abs().max()) if t.numel() else 0.0
    scale = absmax / FP8_MAX if absmax > 0 else 1.0
    return _pack(code, scale, t, (t / scale).to(dtype))


def _decode_plain(t: Tensor, scale: float, out_dtype: torch.dtype) -> Tensor:
    return t.to(out_dtype)


def _decode_fp8(t: Tensor, scale: float, out_dtype: torch.dtype) -> Tensor:
    return (t.float() * scale).to(out_dtype)


ENCODERS = {
    name: partial(_encode_fp8 if name == "fp8" else _encode_plain, code, dtype)
    for name, (code, dtype) in STORAGE.items()
}
DECODERS = {
    code: _decode_fp8 if name == "fp8" else _decode_plain
    for name, (code, _) in STORAGE.items()
}


def encode_tensor(t: Tensor, storage: str) -> bytes:
    """Serialise `t` (any float dtype) in `storage` dtype."""
    return ENCODERS[storage](t)


def decode_tensor(data: bytes, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Inverse of `encode_tensor`, returned as `out_dtype`."""
    magic, code, ndim, _, scale = HEAD.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not a TensorStore value")
    shape = struct.unpack_from(f"<{ndim}I", data, HEAD.size)
    offset = HEAD.size + 4 * ndim
    _, dtype = CODES[code]
    buffer = np.frombuffer(data, dtype=np.uint8, offset=offset).copy()
    t = torch.from_numpy(buffer).view(dtype).reshape(shape)
    return DECODERS[code](t, scale, out_dtype)


class TensorStore:
    """Key -> tensor KVault opened lazily per process (picklable by path)."""

    def __init__(self, path: str, storage: str = "bf16"):
        if storage not in STORAGE:
            raise ValueError(f"storage must be one of {sorted(STORAGE)}")
        self.path = path
        self.storage = storage
        self._encode = ENCODERS[storage]
        self._vault = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_vault"] = None
        return state

    @property
    def vault(self) -> kv.KVault:
        if self._vault is None:
            self._vault = kv.KVault(self.path)
        return self._vault

    def get(self, key: str) -> Tensor | None:
        data = self.vault.get("t:" + key)
        return None if data is None else decode_tensor(data)

    def put(self, key: str, value: Tensor) -> None:
        self.vault["t:" + key] = self._encode(value)
        self.vault["m:" + key] = str(int(value.shape[0])).encode("ascii")

    def __contains__(self, key: str) -> bool:
        return self.vault.exists("t:" + key)

    def lengths(self) -> dict[str, int]:
        """`{key: first dim}` for every stored tensor (loaded once by planners)."""
        out = {}
        for k in self.vault.keys(prefix="m:", limit=1 << 62):
            name = k.decode("utf-8")[2:]
            out[name] = int(self.vault.get(k).decode("ascii"))
        return out

    def flush(self) -> None:
        if self._vault is not None:
            self._vault.checkpoint()
