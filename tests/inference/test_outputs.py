"""Sample output numbering: the next image index in an output directory."""

from pathlib import Path

import pytest

from kohakumoss.inference import outputs
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

NAMES = ("0000_11.png", "0007_5.png", "metadata-0000.json", "12_3.png", "abcd_1.png")


def check_next(next_index, root: Path) -> None:
    """0 for a missing or empty directory; 1 + the highest 4+-digit `NNNN_` index
    otherwise (metadata and short or non-numeric prefixes do not count); the name of
    the next image continues the numbering."""
    assert next_index(root / "does-not-exist") == 0
    empty = root / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    assert next_index(empty) == 0
    full = root / "full"
    full.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        (full / name).write_bytes(b"")
    assert next_index(full) == 8, next_index(full)
    (full / outputs.image_name(next_index(full), 3)).write_bytes(b"")
    assert next_index(full) == 9


def test_next_index(tmp_path):
    check_next(outputs.next_index, tmp_path / "real")
    assert outputs.image_name(8, 3) == "0008_3.png"
    assert outputs.metadata_name(8) == "metadata-0008.json"

    def counting(out):
        return len(list(Path(out).iterdir())) if Path(out).is_dir() else 0

    must_fail(lambda: check_next(counting, tmp_path / "broken"), "a file count")
