"""Caption layer for a danbooru GBC dataset (KohakuGBC export: danbooru post metadata,
`quality` / `selection` scores, one GBC graph per image). Each caption is

    artist, copyright, character, <middle>, quality, era, rating

with the middle part one of `tags` : `nl` : `tags, nl` : `graph` (weights 2 : 2 : 3 : 3).
See docs/concepts/data.md#gbc-captions for every key."""

pre = ["artist", "copyright", "character"]
post = ["quality", "era", "rating"]
middles = [(2, ["tags"]), (2, ["nl"]), (3, ["tags", "nl"]), (3, ["graph"])]

CAPTION = {
    "template": "gbc",
    "meta": {
        "artist": {"field": "artist", "drop": 0.2},
        "copyright": {"field": "copyright", "drop": 0.2},
        "character": {"field": "character", "drop": 0.2},
        # percentile of the normalized score among all live posts; "" = no tag
        "quality": {
            "field": "selection.quality_pct",
            "bins": [
                [0.97, "masterpiece"],
                [0.90, "best quality"],
                [0.75, "great quality"],
                [0.45, ""],
                [0.25, "low quality"],
                [0.10, "bad quality"],
                [0.0, "worst quality"],
            ],
        },
        # upload year, both ends inclusive
        "era": {
            "field": "year",
            "ranges": [
                [2025, 2026, "newest"],
                [2023, 2024, "recent"],
                [2020, 2022, "mid"],
                [2015, 2019, "early"],
                [2005, 2014, "old"],
            ],
        },
        "rating": {
            "field": "rating",
            "map": {
                "general": "safe",
                "sensitive": "sensitive",
                "questionable": "nsfw",
                "explicit": ["nsfw", "explicit"],
            },
        },
    },
    # danbooru general tags or the graph root's tags, one per sample
    "tags": {
        "sources": [
            {"from": "meta:general", "weight": 1.0},
            {"from": "graph:root", "weight": 1.0},
        ],
        "drop": 0.2,
        "shuffle": True,
        # the tagger's rating tags (the rating section says it), and names the
        # artist / copyright / character sections already hold
        "exclude": ["^rating:"],
        "dedupe": ["artist", "copyright", "character"],
    },
    "nl": {"short": 0.3},
    "graph": {"keep": 0.55, "tag_drop": 0.2},
    "formats": [{"weight": w, "sections": pre + mid + post} for w, mid in middles],
    "list_sep": ", ",
    "section_sep": ",\n",
    "text_sep": "\n",
    "underscores": True,
    "fit": {"margin": 0.75, "fallback": pre + ["nl"] + post},
}
