"""TensorStore: KVault files of headered tensors, one writer per file (the main file, or
a `<stem>.r<rank>.kv` shard per online writer). Value and key layout:
docs/concepts/data.md#tensor-store-files."""

import gc
import glob
import os
import struct
import time
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
REFRESH = 5.0


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


def _header(data: bytes) -> tuple[int, float, tuple[int, ...]]:
    """(dtype code, scale, shape) of a stored value."""
    magic, code, ndim, _, scale = HEAD.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not a TensorStore value")
    return code, scale, struct.unpack_from(f"<{ndim}I", data, HEAD.size)


def first_dim(data: bytes) -> int:
    """The first dim (token count) of a stored value, from its header."""
    return int(_header(data)[2][0])


def decode_tensor(data: bytes, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Inverse of `encode_tensor`, returned as `out_dtype`."""
    code, scale, shape = _header(data)
    offset = HEAD.size + 4 * len(shape)
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


def shard_path(path: str, rank: int) -> str:
    """The shard file one writer (`rank`) owns next to the main file `path`."""
    stem = os.path.splitext(path)[0]
    return f"{stem}.r{rank}.kv"


def shard_paths(path: str) -> list[str]:
    """Every shard file present next to `path`, in rank order."""
    stem = os.path.splitext(path)[0]
    found = glob.glob(glob.escape(stem) + ".r*.kv")
    return sorted(found, key=lambda p: int(p[len(stem) + 2 : -3] or 0))


class TensorStore:
    """Key -> tensor: the main KVault file `path` plus the per-writer shard files next
    to it. Reads see the union (main first, then shards by rank); `put` writes only
    `writes` (default: the main file). Every file has one writer; opened lazily per
    process (picklable by path). Keys are 32-hex-digit cache hashes."""

    def __init__(
        self,
        path: str,
        storage: str = "bf16",
        retries: int = 4,
        backoff: float = 0.02,
        writes: str | None = None,
    ):
        if storage not in STORAGE:
            raise ValueError(f"storage must be one of {sorted(STORAGE)}")
        self.path = path
        self.writes = writes or path
        self.storage = storage
        self.retries = retries
        self.backoff = backoff
        self._encode = ENCODERS[storage]
        self._vault = None
        self._readers: dict[str, kv.KVault] = {}
        self._listed = 0.0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_vault"] = None
        state["_readers"] = {}
        state["_listed"] = 0.0
        return state

    def _open(self, path: str) -> kv.KVault:
        return kv.KVault(path, retries=self.retries, backoff_base=self.backoff)

    @property
    def vault(self) -> kv.KVault:
        """The file this store writes."""
        if self._vault is None:
            self._vault = self._open(self.writes)
        return self._vault

    def readers(self) -> list[kv.KVault]:
        """Every file present (main first, then shards), re-listed at most every
        REFRESH seconds."""
        now = time.monotonic()
        if now - self._listed > REFRESH:
            self._listed = now
            for path in [self.path, *shard_paths(self.path)]:
                if path not in self._readers and os.path.exists(path):
                    self._readers[path] = (
                        self.vault if path == self.writes else self._open(path)
                    )
        return list(self._readers.values())

    def _find(self, raw: str) -> bytes | None:
        for vault in self.readers():
            data = vault.get(raw)
            if data is not None:
                return data
        return None

    def get(self, key: str) -> Tensor | None:
        """The tensor under `key` (bf16) from any file, or None."""
        data = self._find("t:" + key)
        return None if data is None else decode_tensor(data)

    def put(self, key: str, value: Tensor) -> None:
        """Store `value` under `key` in the file this store writes."""
        self.vault["t:" + key] = self._encode(value)
        self.vault["m:" + key] = str(int(value.shape[0])).encode("ascii")
        self._readers.setdefault(self.writes, self.vault)

    def __contains__(self, key: str) -> bool:
        return any(v.exists("t:" + key) for v in self.readers())

    def scan_lengths(self) -> dict[str, int]:
        """`{key: first dim}` from every `m:` entry of every file (main wins)."""
        out: dict[str, int] = {}
        for vault in reversed(self.readers()):
            for k in vault.keys(prefix="m:", limit=1 << 62):
                out[k.decode("ascii")[2:]] = int(vault.get(k).decode("ascii"))
        return out

    def write_index(self) -> int:
        """Store the union's length table under `idx:lengths` of the file this store
        writes (an offline / merge step: one process); returns its size."""
        table = LengthTable.from_dict(self.scan_lengths())
        self.vault[INDEX_KEY] = table.to_bytes()
        return len(table)

    def lengths(self) -> LengthTable:
        """The main file's length table (one read) when no shard exists, else a scan
        of the union."""
        main = self._readers.get(self.path) or (
            self._open(self.path) if os.path.exists(self.path) else None
        )
        data = None if main is None or shard_paths(self.path) else main.get(INDEX_KEY)
        if data is None:
            return LengthTable.from_dict(self.scan_lengths())
        return LengthTable.from_bytes(data)

    def flush(self) -> None:
        """Checkpoint the file this store writes."""
        if self._vault is not None:
            self._vault.checkpoint()


def merge_shards(path: str) -> int:
    """Fold every shard next to `path` into the main file (keys the main file lacks),
    rewrite its length index, delete the shards; one process, no writer running.
    Returns the number of entries copied."""
    # Every unreachable handle on the shards is closed before they are deleted.
    gc.collect()
    main = kv.KVault(path)
    copied = 0
    for shard in shard_paths(path):
        vault = kv.KVault(shard)
        for k in vault.keys(prefix="t:", limit=1 << 62):
            if main.exists(k):
                continue
            data = vault.get(k)
            main[k] = data
            main[b"m:" + k[2:]] = str(first_dim(data)).encode("ascii")
            copied += 1
        vault.close()
        del vault
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(shard + suffix):
                os.remove(shard + suffix)
    if main.get(INDEX_KEY) is not None or copied:
        store = TensorStore(path)
        store.write_index()
        store.flush()
    main.checkpoint()
    return copied
