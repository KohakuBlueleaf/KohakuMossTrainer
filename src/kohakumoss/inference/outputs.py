"""Output file names of the sample script: images `<index>_<seed>.png` numbered after
the highest index already in the directory, one `metadata-<first index>.json` per run.
"""

import re
from pathlib import Path

INDEX_RE = re.compile(r"^(\d{4,})_")


def next_index(out: Path) -> int:
    """1 + the highest `NNNN_` image index in `out` (0 for a new directory)."""
    if not Path(out).is_dir():
        return 0
    found = [
        int(m.group(1)) for p in Path(out).iterdir() if (m := INDEX_RE.match(p.name))
    ]
    return max(found) + 1 if found else 0


def image_name(index: int, seed: int) -> str:
    return f"{index:04d}_{seed}.png"


def metadata_name(first: int) -> str:
    return f"metadata-{first:04d}.json"
