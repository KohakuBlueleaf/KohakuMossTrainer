"""A small kohya-layout slice of the KOHAKUMOSS_DATASET dataset and its DATASETS
entry, for the real-weights tests."""

import shutil
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def kohya_slice(src: Path, dst: Path, n: int) -> Path:
    """The first `n` images (with their sidecars) of each of the first two kohya folders
    of `src`, copied into `dst` with the folder names kept."""
    folders = sorted(p for p in src.iterdir() if p.is_dir() and p.name[0].isdigit())[:2]
    for folder in folders:
        out = dst / folder.name
        out.mkdir(parents=True)
        images = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
        )
        for image in images[: n // len(folders)]:
            for sidecar in folder.glob(image.stem + ".*"):
                if sidecar.suffix.lower() in (*IMAGE_SUFFIXES, ".txt", ".json"):
                    shutil.copy2(sidecar, out / sidecar.name)
    return dst


def kohya_source(path: Path) -> dict:
    """The DATASETS entry of a slice (kohya source, artist from the folder name)."""
    return {
        "source": "kohya",
        "path": str(path),
        "repeat_prefix": "first",
        "artist_from_folder": "name",
        "txt_as": "tags",
    }
