"""TensorStore: a KVault of tensors with a small binary header.

Value layout: magic `KMT1`, dtype code u8, ndim u8, pad u16, scale f32, shape u32 x ndim,
then the raw little-endian payload. fp8 values are stored as e4m3 with a per-tensor
fp32 scale `absmax / 448` (computed in fp32). Key `m:<key>` holds the first dim (token
count), written after `t:<key>`; `idx:lengths` holds a packed length table of every
entry (see `LengthTable`). See docs/concepts/data.md.
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
INDEX_KEY = "idx:lengths"
COUNT = struct.Struct("<I")


def _pack(code: int, scale: float, t: Tensor, payload: Tensor) -> bytes:
    raw = payload.view(torch.uint8).numpy().tobytes()
    shape = struct.pack(f"<{t.ndim}I", *t.shape)
    return HEAD.pack(MAGIC, code, t.ndim, 0, scale) + shape + raw


def fp8_scale(t: Tensor) -> Tensor:
    """fp32 scalar scale of a tensor stored as fp8: absmax / 448, or 1 for all-zero."""
    absmax = t.detach().float().abs().amax() if t.numel() else t.new_zeros(())
    return torch.where(absmax > 0, absmax / FP8_MAX, torch.ones_like(absmax)).float()


def _encode_plain(code: int, dtype: torch.dtype, t: Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous()
    return _pack(code, 1.0, t, t.to(dtype))


def _encode_fp8(code: int, dtype: torch.dtype, t: Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous().float()
    scale = fp8_scale(t)
    return _pack(code, float(scale), t, (t / scale).to(dtype))


def _decode_plain(t: Tensor, scale: float, out_dtype: torch.dtype) -> Tensor:
    return t.to(out_dtype)


def _decode_fp8(t: Tensor, scale: float, out_dtype: torch.dtype) -> Tensor:
    return (t.float() * scale).to(out_dtype)


def _stored_plain(dtype: torch.dtype, t: Tensor, out_dtype: torch.dtype) -> Tensor:
    return t.detach().to(dtype).to(out_dtype)


def _stored_fp8(dtype: torch.dtype, t: Tensor, out_dtype: torch.dtype) -> Tensor:
    scale = fp8_scale(t)
    return ((t.detach().float() / scale).to(dtype).float() * scale).to(out_dtype)


ENCODERS = {
    name: partial(_encode_fp8 if name == "fp8" else _encode_plain, code, dtype)
    for name, (code, dtype) in STORAGE.items()
}
DECODERS = {
    code: _decode_fp8 if name == "fp8" else _decode_plain
    for name, (code, _) in STORAGE.items()
}
STORED = {
    name: partial(_stored_fp8 if name == "fp8" else _stored_plain, dtype)
    for name, (_, dtype) in STORAGE.items()
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


def stored_value(
    t: Tensor, storage: str, out_dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    """What `decode_tensor(encode_tensor(t, storage), out_dtype)` returns, computed on
    `t`'s device (bit-identical)."""
    return STORED[storage](t, out_dtype)


class LengthTable:
    """Sorted 16-byte cache keys -> first-dim lengths; compact and cheap to pickle."""

    def __init__(self, keys: np.ndarray | None = None, lens: np.ndarray | None = None):
        self.keys = np.empty(0, dtype="S16") if keys is None else keys
        self.lens = np.empty(0, dtype=np.int32) if lens is None else lens

    @classmethod
    def from_dict(cls, lengths: dict[str, int]) -> "LengthTable":
        items = sorted((bytes.fromhex(k), n) for k, n in lengths.items())
        keys = np.array([k for k, _ in items], dtype="S16")
        lens = np.array([n for _, n in items], dtype=np.int32)
        return cls(keys, lens)

    @classmethod
    def from_bytes(cls, data: bytes) -> "LengthTable":
        (n,) = COUNT.unpack_from(data, 0)
        keys = np.frombuffer(data, dtype="S16", count=n, offset=COUNT.size).copy()
        lens = np.frombuffer(
            data, dtype="<i4", count=n, offset=COUNT.size + 16 * n
        ).astype(np.int32)
        return cls(keys, lens)

    def to_bytes(self) -> bytes:
        return (
            COUNT.pack(len(self.keys))
            + self.keys.tobytes()
            + self.lens.astype("<i4").tobytes()
        )

    def __len__(self) -> int:
        return len(self.keys)

    def get(self, key: str) -> int | None:
        raw = np.array(bytes.fromhex(key), dtype="S16")
        i = int(np.searchsorted(self.keys, raw))
        if i < len(self.keys) and self.keys[i] == raw:
            return int(self.lens[i])
        return None


class TensorStore:
    """Key -> tensor KVault opened lazily per process (picklable by path). Keys are
    32-hex-digit cache hashes."""

    def __init__(
        self,
        path: str,
        storage: str = "bf16",
        retries: int = 4,
        backoff: float = 0.02,
    ):
        if storage not in STORAGE:
            raise ValueError(f"storage must be one of {sorted(STORAGE)}")
        self.path = path
        self.storage = storage
        self.retries = retries
        self.backoff = backoff
        self._encode = ENCODERS[storage]
        self._vault = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_vault"] = None
        return state

    @property
    def vault(self) -> kv.KVault:
        if self._vault is None:
            self._vault = kv.KVault(
                self.path, retries=self.retries, backoff_base=self.backoff
            )
        return self._vault

    def get(self, key: str) -> Tensor | None:
        data = self.vault.get("t:" + key)
        return None if data is None else decode_tensor(data)

    def put(self, key: str, value: Tensor) -> None:
        self.vault["t:" + key] = self._encode(value)
        self.vault["m:" + key] = str(int(value.shape[0])).encode("ascii")

    def __contains__(self, key: str) -> bool:
        return self.vault.exists("t:" + key)

    def scan_lengths(self) -> dict[str, int]:
        """`{key: first dim}` from every `m:` entry (one pass over the metadata keys)."""
        out = {}
        for k in self.vault.keys(prefix="m:", limit=1 << 62):
            name = k.decode("ascii")[2:]
            out[name] = int(self.vault.get(k).decode("ascii"))
        return out

    def write_index(self) -> int:
        """Store the length table of every entry under `idx:lengths`; returns its size."""
        table = LengthTable.from_dict(self.scan_lengths())
        self.vault[INDEX_KEY] = table.to_bytes()
        return len(table)

    def lengths(self) -> LengthTable:
        """The stored length table (one read), or a scan when no index was written."""
        data = self.vault.get(INDEX_KEY)
        if data is None:
            return LengthTable.from_dict(self.scan_lengths())
        return LengthTable.from_bytes(data)

    def flush(self) -> None:
        if self._vault is not None:
            self._vault.checkpoint()
