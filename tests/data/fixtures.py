"""Synthetic on-disk datasets for data tests."""

import io
import json
import os
import tarfile

import numpy as np
from PIL import Image


def png_bytes(w: int, h: int, seed: int, alpha: bool = False) -> bytes:
    rng = np.random.default_rng(seed)
    mode = "RGBA" if alpha else "RGB"
    arr = rng.integers(0, 256, size=(h, w, len(mode)), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr, mode).save(buffer, format="PNG")
    return buffer.getvalue()


def write(path: str, data: bytes | str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)


def make_kohya(root: str) -> dict[str, int]:
    """Folders `3_2_alpha` (2 images), `1_beta` (1 image), `7_gamma` (1 image + an
    orphan sidecar); returns {folder: image count}."""
    layout = {"3_2_alpha": 2, "1_beta": 1, "7_gamma": 1}
    seed = 0
    for folder, n in layout.items():
        for i in range(n):
            stem = os.path.join(root, folder, f"img{i}")
            write(stem + ".png", png_bytes(64 + 16 * i, 48, seed))
            write(stem + ".txt", f"tag_a, tag\\(b\\), {folder}_{i}")
            seed += 1
    write(os.path.join(root, "7_gamma", "orphan.txt"), "no image here")
    return layout


def make_folder(root: str) -> int:
    write(os.path.join(root, "a", "x.png"), png_bytes(32, 32, 1))
    write(os.path.join(root, "a", "x.caption"), "a red square")
    write(os.path.join(root, "b", "y.png"), png_bytes(48, 32, 2, alpha=True))
    write(
        os.path.join(root, "b", "y.json"),
        json.dumps({"tags": ["t1", "t2"], "artist": "someone", "layout": {"k": 1}}),
    )
    return 2


def make_jsonl(root: str) -> str:
    write(os.path.join(root, "img", "p.png"), png_bytes(40, 24, 3))
    write(os.path.join(root, "img", "q.png"), png_bytes(24, 40, 4))
    lines = [
        {"image": "img/p.png", "tags": "x, y", "weight": 2},
        {"file": "img/q.png", "caption": "tall image", "rating": "g"},
    ]
    path = os.path.join(root, "meta.jsonl")
    write(path, "\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def make_tar(path: str, n: int = 3) -> list[bytes]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = []
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            data = png_bytes(32, 16 + 16 * i, 10 + i)
            images.append(data)
            for name, payload in (
                (f"k{i:03d}.png", data),
                (f"k{i:03d}.txt", f"wd_{i}, other".encode()),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return images
