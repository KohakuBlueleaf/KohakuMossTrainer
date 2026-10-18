"""Image shards: `NNNNN.bin` (concatenated encoded images) + `NNNNN.idx` (32-byte records).

Record layout, little-endian: offset u64, size u32, w u16, h u16, codec u8, flags u8,
6 bytes pad, content hash prefix u64. See docs/concepts/data.md.
"""

import os
import struct
from dataclasses import dataclass

import numpy as np

RECORD = struct.Struct("<QIHHBB6xQ")
CODECS = {"jpeg": 1, "png": 2, "webp": 3, "jxl": 4, "avif": 5, "bmp": 6, "other": 0}
FLAG_ALPHA = 1
SHARD_BYTES = 1 << 30


@dataclass(frozen=True)
class ShardRecord:
    """One image inside a shard."""

    offset: int
    size: int
    w: int
    h: int
    codec: int
    flags: int
    hash64: int


def codec_of(name: str) -> int:
    """The codec id of a file name's extension."""
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    match ext:
        case "jpg" | "jpeg":
            return CODECS["jpeg"]
        case "png" | "webp" | "jxl" | "avif" | "bmp":
            return CODECS[ext]
        case _:
            return CODECS["other"]


class ShardWriter:
    """Appends images to `dir/shards/NNNNN.{bin,idx}`; rolls over at `shard_bytes`."""

    def __init__(self, root: str, shard_bytes: int = SHARD_BYTES):
        self.dir = os.path.join(root, "shards")
        os.makedirs(self.dir, exist_ok=True)
        self.shard_bytes = shard_bytes
        self.shard = -1
        self._bin = None
        self._idx = None
        self._offset = 0
        self._roll()

    def _roll(self) -> None:
        self.close()
        self.shard += 1
        stem = os.path.join(self.dir, f"{self.shard:05d}")
        self._bin = open(stem + ".bin", "wb")
        self._idx = open(stem + ".idx", "wb")
        self._offset = 0

    def add(
        self, data: bytes, w: int, h: int, codec: int, alpha: bool, hash64: int
    ) -> str:
        """Write one image; returns its locator `shard:index`."""
        if self._offset and self._offset + len(data) > self.shard_bytes:
            self._roll()
        index = self._idx.tell() // RECORD.size
        flags = FLAG_ALPHA if alpha else 0
        self._idx.write(
            RECORD.pack(self._offset, len(data), w, h, codec, flags, hash64)
        )
        self._bin.write(data)
        self._offset += len(data)
        return f"{self.shard}:{index}"

    def close(self) -> None:
        for handle in (self._bin, self._idx):
            if handle is not None:
                handle.close()
        self._bin = self._idx = None


class ShardReader:
    """Random access into a shard directory; file handles opened lazily per process."""

    def __init__(self, root: str):
        self.dir = os.path.join(root, "shards")
        self._index: dict[int, np.ndarray] = {}
        self._handles: dict[int, object] = {}

    def __getstate__(self):
        return {"dir": self.dir}

    def __setstate__(self, state):
        self.dir = state["dir"]
        self._index = {}
        self._handles = {}

    def _records(self, shard: int) -> np.ndarray:
        if shard not in self._index:
            path = os.path.join(self.dir, f"{shard:05d}.idx")
            raw = np.fromfile(path, dtype=np.uint8).reshape(-1, RECORD.size)
            self._index[shard] = raw
        return self._index[shard]

    def record(self, locator: str) -> ShardRecord:
        """The index record at `locator` (`shard:index`)."""
        shard, index = (int(p) for p in locator.split(":"))
        raw = bytes(self._records(shard)[index])
        return ShardRecord(*RECORD.unpack(raw))

    def read(self, locator: str) -> bytes:
        """The encoded image bytes at `locator` (`shard:index`)."""
        shard = int(locator.split(":", 1)[0])
        rec = self.record(locator)
        handle = self._handles.get(shard)
        if handle is None:
            handle = open(os.path.join(self.dir, f"{shard:05d}.bin"), "rb")
            self._handles[shard] = handle
        handle.seek(rec.offset)
        return handle.read(rec.size)
