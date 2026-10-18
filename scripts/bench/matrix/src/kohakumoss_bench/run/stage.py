"""Weights staging on the host: copy the model (and the fp16-fix file) once to fast local
storage (NVMe, or /dev/shm where RAM allows) and load every run from there; a source
`hf:<repo>` downloads the repository into the stage directory instead. Files already
staged with the same size are kept, so a second bench start stages nothing.
See docs/performance/bench-matrix.md#host-runner.
"""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

HF = "hf:"


def _same(src: Path, dst: Path) -> bool:
    return dst.is_file() and dst.stat().st_size == src.stat().st_size


def copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    """Copy `src` (a file or a directory) to `dst`; (files copied, files kept)."""
    copied = kept = 0
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for f in files:
        target = dst if src.is_file() else dst / f.relative_to(src)
        if _same(f, target):
            kept += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        shutil.copyfile(f, part)
        part.replace(target)
        copied += 1
    return copied, kept


def stage(entry: dict) -> dict:
    """Stage one weights entry {"source", "stage"}; the report of what happened."""
    source, target = entry.get("source", ""), entry.get("stage", "")
    if not source or not target or source == target:
        return {"source": source, "stage": target or source, "copied": 0, "kept": 0}
    if source.startswith(HF):
        path = snapshot_download(repo_id=source[len(HF) :], local_dir=target)
        return {"source": source, "stage": path, "downloaded": True}
    copied, kept = copy_tree(Path(source), Path(target))
    return {"source": source, "stage": target, "copied": copied, "kept": kept}
