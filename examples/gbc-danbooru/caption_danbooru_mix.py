"""Caption layer for a danbooru GBC dataset with an artist-heavy share: of every caption

    10 %  empty (TEXT_DROP)
    40 %  artist heavy: artist, quality, era, rating (posts with an artist only)
    30 %  artist, copyright, character, tags | tags, nl | nl, quality, era, rating
    20 %  artist, copyright, character, graph, quality, era, rating

Every drop rate is a range drawn per caption, so some captions keep every item. A post
without an artist draws among the other formats. Builds on caption_danbooru.py's
metadata rules. See docs/concepts/data.md#gbc-captions."""

import runpy
from pathlib import Path

base = runpy.run_path(str(Path(__file__).with_name("caption_danbooru.py")))["CAPTION"]
pre = ["artist", "copyright", "character"]
post = ["quality", "era", "rating"]

TEXT_DROP = 0.1
CAPTION = {
    **base,
    "meta": {
        **base["meta"],
        "artist": {"field": "artist", "drop": [0.0, 0.3]},
        "copyright": {"field": "copyright", "drop": [0.0, 0.4]},
        "character": {"field": "character", "drop": [0.0, 0.4]},
    },
    "tags": {**base["tags"], "drop": [0.0, 0.4]},
    "graph": {"keep": [0.35, 0.8], "tag_drop": [0.0, 0.4]},
    "formats": [
        {
            "weight": 40,
            "sections": ["artist"] + post,
            "requires": ["artist"],
            "drop": {"artist": 0.0},
        },
        {"weight": 10, "sections": pre + ["tags"] + post},
        {"weight": 10, "sections": pre + ["tags", "nl"] + post},
        {"weight": 10, "sections": pre + ["nl"] + post},
        {"weight": 20, "sections": pre + ["graph"] + post},
    ],
}
