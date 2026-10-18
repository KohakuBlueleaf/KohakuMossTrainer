"""Per-row caption-source lengths (characters), the planning columns the text-token
estimate is computed from without reading any row."""

import json

from kohakumoss_data.types import CatalogRow

CHAR_FIELDS = ("tags_chars", "nl_chars", "artist_chars", "rating_chars", "json_chars")


def _chars(value) -> int:
    match value:
        case None | "":
            return 0
        case list() | tuple():
            return len(", ".join(str(v) for v in value))
        case _:
            return len(str(value))


def text_chars(row: CatalogRow) -> tuple[int, int, int, int, int]:
    """(tags, caption_nl, artist, rating, json) character counts of `row`, in
    CHAR_FIELDS order; tags as joined by ", ", json as its compact dump."""
    meta = len(json.dumps(row.json, ensure_ascii=False)) if row.json else 0
    return (
        _chars(row.tags),
        _chars(row.caption_nl),
        _chars(row.artist),
        _chars(row.rating),
        meta,
    )
