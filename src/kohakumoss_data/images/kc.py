"""KohakuClip image shards read by key: an index on disk (`build_index`, `KcIndex`:
key -> shard, byte offset, size, width, height) and `KcStore`, which reads one image's
bytes with one ranged read from a local shard directory or an HTTP root (e.g. `rclone
serve http` over the S3 bucket). See docs/concepts/data.md#kohakuclip-image-shards."""

import json
import os
import struct
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import requests

from kohakumoss_common.retry import retrying

TIMEOUT = (10.0, 60.0)
READ_DEADLINE = 240.0
BACKOFF = 2.0
BACKOFF_MAX = 30.0
MAGIC = b"KCIMG1\0\0"
INDEX_NAME = b"__index__.bin"
RECORD = np.dtype(
    [
        ("off", "<u8"),
        ("size", "<u4"),
        ("w", "<u4"),
        ("h", "<u4"),
        ("codec", "u1"),
        ("color", "u1"),
        ("pad", "u1", (2,)),
    ]
)
COLUMNS = {"key": np.int64, "shard": np.int32, "off": np.int64, "size": np.int64}
COLUMNS |= {"w": np.int32, "h": np.int32}
SHARDS = "shards.json"
_EOCD, _EOCD64, _LOCATOR, _CENTRAL = 0x06054B50, 0x06064B50, 0x07064B50, 0x02014B50
_U32 = 0xFFFFFFFF
TAIL = 22 + 65535 + 20


class ShortRead(OSError):
    """A ranged read returned fewer bytes than asked (a cut transfer)."""


TRANSIENT = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.HTTPError,
    ShortRead,
)


def _transient(err: BaseException) -> bool:
    """Every TRANSIENT failure except an HTTP status below 500."""
    if isinstance(err, requests.HTTPError):
        return err.response is not None and err.response.status_code >= 500
    return True


def _checked(call, *args, **kw):
    """`call(*args, **kw)` raising for an HTTP error status."""
    got = call(*args, **kw)
    got.raise_for_status()
    return got


class LocalShard:
    """Ranged reads of one local shard file (thread-safe: one handle, one lock)."""

    def __init__(self, path: str):
        self.file = open(path, "rb")  # noqa: SIM115 (open while the store lives)
        self.size = os.fstat(self.file.fileno()).st_size
        self.lock = threading.Lock()

    def read(self, offset: int, length: int) -> bytes:
        with self.lock:
            self.file.seek(offset)
            return self.file.read(length)

    def close(self) -> None:
        self.file.close()


class HttpShard:
    """Ranged reads of one shard behind an HTTP URL (one GET with a Range header). A
    failed read (connection error, timeout, 5xx, cut transfer) is retried with backoff
    while the next attempt starts within `deadline` seconds of the first (READ_DEADLINE:
    the read stays well inside DIST_TIMEOUT); a 4xx or a server without ranged reads
    raises at once. `attempts` also caps the count (None: the deadline alone)."""

    def __init__(
        self,
        url: str,
        session: requests.Session,
        deadline: float = READ_DEADLINE,
        attempts: int | None = None,
        backoff: float = BACKOFF,
    ):
        self.url = url
        self.session = session
        self.retry = partial(
            retrying,
            transient=TRANSIENT,
            what=url,
            deadline=deadline,
            attempts=attempts,
            backoff=backoff,
            backoff_max=BACKOFF_MAX,
            is_transient=_transient,
        )
        head = self.retry(partial(_checked, session.head, url, timeout=TIMEOUT))
        self.size = int(head.headers["Content-Length"])

    def _get(self, rng: dict, length: int) -> bytes:
        with self.session.get(
            self.url, headers=rng, timeout=TIMEOUT, stream=True
        ) as got:
            got.raise_for_status()
            if got.status_code != 206:
                raise OSError(f"{self.url}: no ranged read (HTTP {got.status_code})")
            data = got.content
        if len(data) != length:
            raise ShortRead(f"{self.url}: {len(data)} bytes for {length} asked")
        return data

    def read(self, offset: int, length: int) -> bytes:
        rng = {"Range": f"bytes={offset}-{offset + length - 1}"}
        return self.retry(partial(self._get, rng, length))

    def close(self) -> None:
        """Nothing to release (the session is shared)."""


def open_shard(root: str, name: str, session: requests.Session | None = None):
    """A ranged reader of shard `name` under `root` (a directory or an http(s) URL)."""
    if root.startswith(("http://", "https://")):
        return HttpShard(f"{root.rstrip('/')}/{name}", session or requests.Session())
    return LocalShard(os.path.join(root, name))


def _directory(shard) -> tuple[int, int]:
    """(central directory size, offset) of a zip, zip64 aware."""
    length = min(shard.size, TAIL)
    tail = shard.read(shard.size - length, length)
    at = tail.rfind(struct.pack("<I", _EOCD))
    if at < 0:
        raise ValueError("not a zip archive")
    if at >= 20 and struct.unpack_from("<I", tail, at - 20)[0] == _LOCATOR:
        (eocd64,) = struct.unpack_from("<Q", tail, at - 20 + 8)
        record = shard.read(eocd64, 56)
        if struct.unpack_from("<I", record)[0] != _EOCD64:
            raise ValueError("bad zip64 end of central directory")
        cd_size, cd_offset = struct.unpack_from("<QQ", record, 40)
        return cd_size, cd_offset
    cd_size, cd_offset = struct.unpack_from("<II", tail, at + 12)
    return cd_size, cd_offset


def member_names(cd: bytes) -> list[str]:
    """Every member name of a central directory, in order."""
    names, at = [], 0
    while at + 46 <= len(cd) and struct.unpack_from("<I", cd, at)[0] == _CENTRAL:
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", cd, at + 28)
        names.append(cd[at + 46 : at + 46 + name_len].decode())
        at += 46 + name_len + extra_len + comment_len
    return names


def shard_keys(shard) -> tuple[np.ndarray, np.ndarray]:
    """(image keys, index records) of a KohakuClip image shard: each image member
    `<key>.<ext>` (an integer key) and its record, in image order."""
    cd_size, cd_offset = _directory(shard)
    cd = shard.read(cd_offset, cd_size)
    names = [n for n in member_names(cd) if n.encode() != INDEX_NAME]
    records = _records(shard, cd, cd_size)
    if len(names) != len(records):
        raise ValueError(f"{len(names)} image members for {len(records)} records")
    keys = np.array([int(n.rsplit(".", 1)[0]) for n in names], dtype=np.int64)
    return keys, records


def shard_records(shard) -> np.ndarray:
    """The image index records (`RECORD`) of a KohakuClip image shard, read from its
    last member `__index__.bin` (the member order is the image order)."""
    cd_size, cd_offset = _directory(shard)
    return _records(shard, shard.read(cd_offset, cd_size), cd_size)


def _records(shard, cd: bytes, cd_size: int) -> np.ndarray:
    """The records of `__index__.bin`, the last entry of central directory `cd`."""
    for extra in (0, 12):
        entry_len = 46 + len(INDEX_NAME) + extra
        entry = cd[cd_size - entry_len :]
        sig = struct.unpack_from("<I", entry)[0]
        name_len, got_extra, comment = struct.unpack_from("<HHH", entry, 28)
        if (sig, name_len, got_extra, comment) != (_CENTRAL, len(INDEX_NAME), extra, 0):
            continue
        if entry[46 : 46 + len(INDEX_NAME)] != INDEX_NAME:
            continue
        (header,) = struct.unpack_from("<I", entry, 42)
        if header == _U32:
            (header,) = struct.unpack_from("<Q", entry, 46 + len(INDEX_NAME) + 4)
        break
    else:
        raise ValueError("no image index as the last member")
    local = shard.read(header, 30)
    name_len, extra_len = struct.unpack_from("<HH", local, 26)
    start = header + 30 + name_len + extra_len
    head = shard.read(start, 16)
    if head[:8] != MAGIC:
        raise ValueError("bad image index magic")
    (head_len,) = struct.unpack_from("<Q", head, 8)
    info = json.loads(shard.read(start + 16, head_len))
    at = start + 16 + head_len + (-(16 + head_len) % 8)
    count = int(info["images"])
    return np.frombuffer(shard.read(at, count * RECORD.itemsize), dtype=RECORD)


def _members(keys: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Which `keys` are in the sorted array `wanted` (binary search, no sort of
    `wanted` per call)."""
    at = np.searchsorted(wanted, keys).clip(max=len(wanted) - 1)
    return wanted[at] == keys if len(wanted) else np.zeros(len(keys), dtype=bool)


def build_index(
    root: str,
    shards: list[str],
    out: str | Path,
    keep: set[int] | None = None,
    workers: int = 32,
) -> int:
    """Write the index of `shards` under `root` to directory `out`: one array per
    column (`COLUMNS`, sorted by key) and the shard names, from each shard's member
    names and index records (`shard_keys`); `keep` limits the keys. A key in two
    shards keeps its first shard. Returns the rows written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = None if keep is None else np.sort(np.fromiter(keep, dtype=np.int64))

    def one(i: int):
        keys, rec = shard_keys(open_shard(root, shards[i]))
        mask = slice(None) if wanted is None else _members(keys, wanted)
        r = rec[mask]
        return keys[mask], np.full(len(r), i, np.int32), r

    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(one, range(len(shards))))
    key = np.concatenate([p[0] for p in parts])
    order = np.argsort(key, kind="stable")
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    order = order[first]
    cols = {
        "key": key,
        "shard": np.concatenate([p[1] for p in parts]),
        "off": np.concatenate([p[2]["off"] for p in parts]).astype(np.int64),
        "size": np.concatenate([p[2]["size"] for p in parts]).astype(np.int64),
        "w": np.concatenate([p[2]["w"] for p in parts]).astype(np.int32),
        "h": np.concatenate([p[2]["h"] for p in parts]).astype(np.int32),
    }
    for name, values in cols.items():
        np.save(out / f"{name}.npy", values[order].astype(COLUMNS[name]))
    (out / SHARDS).write_text(json.dumps(shards), encoding="utf-8")
    return len(order)


class KcIndex:
    """A `build_index` directory, memory-mapped: `find(key)` -> row or -1."""

    def __init__(self, path: str | Path):
        path = Path(path)
        self.cols = {c: np.load(path / f"{c}.npy", mmap_mode="r") for c in COLUMNS}
        self.shards = json.loads((path / SHARDS).read_text(encoding="utf-8"))

    def find(self, key: int) -> int:
        keys = self.cols["key"]
        i = int(np.searchsorted(keys, key))
        return i if i < len(keys) and int(keys[i]) == key else -1

    def size_of(self, row: int) -> tuple[int, int]:
        return int(self.cols["w"][row]), int(self.cols["h"][row])


class KcStore:
    """Image bytes by key from the shards an index names (`root`: a directory or an
    http(s) URL); shard readers open on first use, per process, at most `OPEN` at once
    (the least recently used closes)."""

    OPEN = 256

    def __init__(self, index: str | Path, root: str):
        self.index_path = str(index)
        self.root = root
        self._index = None
        self._open: OrderedDict[int, object] = OrderedDict()
        self._session = None
        self._lock = threading.Lock()

    def __getstate__(self):
        return {"index_path": self.index_path, "root": self.root}

    def __setstate__(self, state):
        self.__init__(state["index_path"], state["root"])

    @property
    def index(self) -> KcIndex:
        if self._index is None:
            self._index = KcIndex(self.index_path)
        return self._index

    def _shard(self, i: int):
        with self._lock:
            shard = self._open.get(i)
            if shard is not None:
                self._open.move_to_end(i)
                return shard
            if self._session is None and self.root.startswith(("http://", "https://")):
                self._session = requests.Session()
            shard = open_shard(self.root, self.index.shards[i], self._session)
            self._open[i] = shard
            if len(self._open) > self.OPEN:
                self._open.popitem(last=False)[1].close()
            return shard

    def read(self, key: int) -> bytes:
        """The encoded bytes of image `key` (KeyError when the index lacks it)."""
        row = self.index.find(key)
        if row < 0:
            raise KeyError(f"image {key} is not in {self.index_path}")
        cols = self.index.cols
        shard = self._shard(int(cols["shard"][row]))
        return shard.read(int(cols["off"][row]), int(cols["size"][row]))
