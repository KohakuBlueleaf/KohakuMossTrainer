"""Shared source helpers: image extensions, sidecar parsing, fingerprints."""

import json
import os

from kohakumoss_data.caption.tags import split_tags
from kohakumoss_data.format.header import probe_image
from kohakumoss_data.seeds import content_hash
from kohakumoss_data.types import CatalogRow

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jxl", ".avif")


def is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def file_fingerprint(path: str) -> str:
    """Header-only identity of a file: size and mtime (cache key part)."""
    st = os.stat(path)
    return f"stat:{st.st_size}:{st.st_mtime_ns}"


def read_text(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().strip()


def apply_sidecars(row: CatalogRow, stem: str, txt_as: str) -> None:
    """Fill caption fields from `stem.txt`, `stem.caption`, `stem.json` next to an image."""
    txt = read_text(stem + ".txt")
    if txt is not None:
        match txt_as:
            case "tags":
                row.tags = split_tags(txt)
            case "nl":
                row.caption_nl = txt
            case _:
                raise ValueError(f"txt_as must be 'tags' or 'nl', got {txt_as}")
    caption = read_text(stem + ".caption")
    if caption is not None:
        row.caption_nl = caption
    meta = read_text(stem + ".json")
    if meta is not None:
        apply_json(row, json.loads(meta))


def apply_json(row: CatalogRow, meta: dict) -> None:
    """Map a metadata dict onto row fields; unknown keys stay in `row.json`."""
    meta = dict(meta)
    tags = meta.pop("tags", None)
    if tags is not None:
        row.tags = split_tags(tags) if isinstance(tags, str) else list(tags)
    for key in ("caption", "text", "caption_nl"):
        if key in meta:
            row.caption_nl = meta.pop(key)
    for key in ("artist", "rating"):
        if key in meta:
            setattr(row, key, meta.pop(key))
    if meta:
        row.json = {**(row.json or {}), **meta}


def image_row(path: str, locator: str, group: str, hash_content: bool) -> CatalogRow:
    """A CatalogRow for an image file, from its header (and optionally its bytes)."""
    w, h, alpha = probe_image(path)
    row = CatalogRow(id=0, locator=locator, w=w, h=h, has_alpha=alpha, group=group)
    if hash_content:
        with open(path, "rb") as handle:
            row.extra["hash"] = content_hash(handle.read())
    else:
        row.extra["hash"] = file_fingerprint(path)
    return row


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()
