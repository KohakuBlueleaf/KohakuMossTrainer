"""Grid prompts from the training data: the `artists` most frequent artists of the
catalog, `per_artist` of their rows each (seeded draw), every caption rendered by the
training caption template (variant 0) exactly as training sees it. Rows without an
artist count as one group "". See docs/performance/bench-matrix.md#host-runner.
"""

from collections import Counter

import numpy as np

from kohakumoss_data import DataConfig, MossDataset


def grid_prompts(
    knobs: dict, artists: int, per_artist: int, seed: int = 0
) -> list[str]:
    """Captions of the data block in `knobs` (training knob names), no caches."""
    cfg = DataConfig.from_knobs(
        {**knobs, "LATENT_CACHE": "off", "TEXT_CACHE": "off", "TEXT_DROP": 0.0}
    )
    dataset = MossDataset(cfg, writer=True)
    catalog = dataset.catalog
    ids = catalog.live_ids()
    by_artist: dict[str, list[int]] = {}
    for i in ids:
        by_artist.setdefault(catalog[int(i)].artist or "", []).append(int(i))
    order = [
        a for a, _ in Counter({a: len(v) for a, v in by_artist.items()}).most_common()
    ]
    rng = np.random.default_rng(seed)
    prompts = []
    for artist in order[:artists]:
        rows = by_artist[artist]
        picks = rng.choice(len(rows), size=min(per_artist, len(rows)), replace=False)
        prompts += [dataset.caption(rows[int(p)], 0) for p in sorted(picks)]
    return prompts
