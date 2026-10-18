"""File helpers without torch: the atomic JSON write every package uses, a JSON read
that tolerates a missing or partial file, and a safetensors metadata read."""

import itertools
import json
import os
import threading
import time
from pathlib import Path

REPLACE_TRIES = 6
REPLACE_WAIT_S = 0.02
_TMP_IDS = itertools.count()


def temp_name(path: Path) -> Path:
    """A temporary file next to `path`, unique per process, thread and write."""
    tag = f"{os.getpid()}.{threading.get_ident()}.{next(_TMP_IDS)}"
    return path.with_name(f"{path.name}.{tag}.tmp")


def replace(tmp: Path, path: Path) -> None:
    """`os.replace(tmp, path)`, retried while another process holds `path` open (a
    PermissionError on Windows)."""
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_WAIT_S * 2**attempt)


def write_json_atomic(
    path: str | Path, obj, indent: int | None = 2, sort_keys: bool = False
) -> None:
    """Write `obj` as UTF-8 JSON to a temporary file unique per process, thread and
    write (`temp_name`), then replace `path` with it (`replace`): readers see the old
    file or the new one, never a partial one; the temporary file is removed when the
    write fails. The parent directory is created. Raises OSError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_name(path)
    try:
        tmp.write_text(json.dumps(obj, indent=indent, sort_keys=sort_keys), "utf-8")
        replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict | None:
    """The UTF-8 JSON in `path`; None when it is missing, unreadable or not JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    """The `__metadata__` string map of a safetensors file (empty when it has none);
    reads the header only."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    return dict(header.get("__metadata__") or {})
