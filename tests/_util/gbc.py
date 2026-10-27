"""A small GBC dataset on disk (KohakuGBC's DatasetWriter): typed graphs with a root, two
boxed entities and a nested part, danbooru-style metadata. Model free."""

import io
import json

import numpy as np
from PIL import Image

try:
    from kohakugbc_data.dataset import DatasetWriter
except ImportError:
    DatasetWriter = None

ROOT_LONG = "A woman stands in a park. She smiles at the viewer."
ROOT_SHORT = "A woman in a park."


def _box(left, top, right, bottom):
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "confidence": None,
    }


def _edge(source, text, target):
    return {"source": source, "text": text, "target": target}


def _vertex(vid, label, box, descs, outs=(), ins=(), tags=None):
    v = {
        "vertex_id": vid,
        "bbox": box,
        "label": label,
        "descs": [{"label": k, "text": t} for k, t in descs],
        "in_edges": list(ins),
        "out_edges": list(outs),
        "sub_masks": [],
        "super_masks": [],
    }
    if tags is not None:
        v["tags"] = tags
    return v


def graph_json(i: int, w: int, h: int) -> str:
    """One typed GBC graph: [image] -> woman (-> hair), tree."""
    to_woman, to_tree = _edge("", "woman", "w"), _edge("", "tree", "t")
    to_hair = _edge("w", "hair", "h")
    vertices = [
        _vertex(
            "",
            "image",
            _box(0.0, 0.0, 1.0, 1.0),
            [("detail", ROOT_LONG), ("short", ROOT_SHORT)],
            outs=[to_woman, to_tree],
            tags={
                "1girl": 0.9,
                "outdoors": 0.8,
                "faint_tag": 0.1,
                "rating:g": 0.9,
                "some_artist": 0.9,
            },
        ),
        _vertex(
            "w",
            "entity",
            _box(0.1, 0.2, 0.5, 0.9),
            [("detail", f"A woman {i} in a red coat. She waves.")],
            outs=[to_hair],
            ins=[to_woman],
            tags={"red_coat": 0.9},
        ),
        _vertex(
            "h",
            "entity",
            _box(0.15, 0.2, 0.3, 0.4),
            [("detail", "Long black hair.")],
            ins=[to_hair],
        ),
        _vertex(
            "t",
            "entity",
            _box(0.8, 0.0, 1.0, 1.0),
            [("detail", "A tall tree.")],
            ins=[to_tree],
        ),
    ]
    return json.dumps(
        {
            "vertices": vertices,
            "img_url": None,
            "img_path": f"{i}.webp",
            "original_caption": None,
            "short_caption": None,
            "detail_caption": None,
            "img_size": [w, h],
            "mask_inside_threshold": 0.85,
        }
    )


def meta(i: int) -> dict:
    return {
        "general": ["long_hair", "smile", f"tag_{i}"],
        "artist": ["some_artist"],
        "copyright": [],
        "character": ["girl_(series)"],
        "rating": ["general", "sensitive", "questionable", "explicit"][i % 4],
        "year": 2014 + i,
        "selection": {"quality_pct": 0.98 - 0.1 * i},
    }


def webp(i: int, w: int, h: int) -> bytes:
    pixels = np.random.default_rng(i).integers(0, 255, (h, w, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, "WEBP", quality=80)
    return buf.getvalue()


def write_dataset(
    root, n: int = 4, per_shard: int = 2, w: int = 64, h: int = 32
) -> None:
    """`n` samples (keys "100".."100+n-1") in shards of `per_shard`."""
    with DatasetWriter(root, per_shard=per_shard, header={"name": "test"}) as out:
        for i in range(n):
            out.add(
                str(100 + i), webp(i, w, h), w, h, "webp", graph_json(i, w, h), meta(i)
            )
