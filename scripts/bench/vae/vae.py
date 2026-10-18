"""VAE dtype bench on real weights and real images: precision of encode (latents) and
decode (pixels) in bf16 / fp16 against the fp32 VAE (error stats, non-finite = overflow),
and ms per image / peak allocated GiB per dtype x {encode, decode} x area x {plain,
tiled} x {eager, channels_last, channels_last + compile} x batch. Writes
runs/bench/vae/<gpu>.json. See docs/internals/offload.md#vae.

    python scripts/bench/vae/vae.py --weights <root>/vae --images <dir> --device cuda:0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kohakumoss.encoders.vae import DTYPES, FluxVae
from kohakumoss_kernels.error_stats import stats
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.hardware.memory import device_used, window

AREAS = {"1MP": 1024, "2MP": 1440}
ACCELS = {"eager": (False, False), "cl": (True, False), "cl_compile": (True, True)}


def load_images(folder: str, n: int, side: int) -> list[np.ndarray]:
    """The first `n` images of `folder`, centre-cropped and resized to side x side RGB."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in exts)[:n]
    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        s = min(im.size)
        left, top = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((side, side), Image.LANCZOS)
        out.append(np.asarray(im))
    return out


def _load(args, dtype: torch.dtype, accel: str, tiled: bool) -> FluxVae:
    vae = FluxVae.from_pretrained(args.weights, args.device, dtype)
    if tiled:
        for name in ("enable_tiling", "enable_slicing"):
            getattr(vae.vae, name, lambda: None)()
    vae.accelerate(*ACCELS[accel])
    return vae


def _summary(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in ("finite",)]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["finite"] = all(r["finite"] for r in rows)
    out["worst_cos"] = min(r["cos"] for r in rows)
    return out


def precision(args) -> list[dict]:
    """Encode / decode stats of every dtype against fp32, per area, over the images."""
    rows = []
    names = [d for d in args.dtypes.split(",") if d != "fp32"]
    for area, side in AREAS.items():
        images = load_images(args.images, args.n, side)
        hw = torch.tensor([[side // 16, side // 16]] * len(images))
        ref = _load(args, torch.float32, "eager", False)
        ref_lat = [ref.encode_images([im], hw[:1]) for im in images]
        ref_pix = [ref.decode(_grid(z, side)) for z in ref_lat]
        del ref
        torch.cuda.empty_cache()
        for name in names:
            vae = _load(args, DTYPES[name], "eager", False)
            enc = [
                stats(vae.encode_images([im], hw[:1]), r)
                for im, r in zip(images, ref_lat)
            ]
            dec = [
                stats(vae.decode(_grid(z, side)), p) for z, p in zip(ref_lat, ref_pix)
            ]
            for what, st in (("encode", enc), ("decode", dec)):
                rows.append({"dtype": name, "area": area, "what": what, **_summary(st)})
                print("PREC", json.dumps(rows[-1]), flush=True)
            del vae
            torch.cuda.empty_cache()
    return rows


def _grid(packed: torch.Tensor, side: int) -> torch.Tensor:
    """Packed latents [h*w, 128] of one image -> [1, 128, h, w]."""
    h = side // 16
    return packed.transpose(0, 1).reshape(1, 128, h, h)


def _timed(fn, dev) -> float:
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize(dev)
    return time.perf_counter() - t0


def speed(args, name: str, accel: str) -> list[dict]:
    """ms / image and peak GiB of one dtype x acceleration over areas, tiling, batches."""
    dev = torch.device(args.device)
    rows = []
    for tiled in (False, True):
        if tiled and accel != "eager":
            continue
        vae = _load(args, DTYPES[name], accel, tiled)
        for area, side in AREAS.items():
            images = load_images(args.images, args.speed_n, side)
            for batch in [int(b) for b in args.batches.split(",")]:
                vae.batch = batch
                hw = torch.tensor([[side // 16, side // 16]] * len(images))
                lat = vae.encode_images(images, hw)
                grid = lat.reshape(len(images), -1, 128).transpose(1, 2)
                grid = grid.reshape(len(images), 128, side // 16, side // 16)

                def decode_all():
                    for lo in range(0, len(images), batch):
                        vae.decode(grid[lo : lo + batch])

                for what, fn in (
                    ("encode", lambda: vae.encode_images(images, hw)),
                    ("decode", decode_all),
                ):
                    row = {
                        "dtype": name,
                        "accel": accel,
                        "tiled": tiled,
                        "area": area,
                        "batch": batch,
                        "what": what,
                    }
                    try:
                        fn()
                        with window(dev) as w:
                            times = [_timed(fn, dev) for _ in range(args.iters)]
                    except torch.OutOfMemoryError:
                        row["oom"] = True
                        torch.cuda.empty_cache()
                    else:
                        median = sorted(times)[len(times) // 2]
                        row["ms_per_image"] = 1e3 * median / len(images)
                        row["peak_GiB"] = w.alloc / 2**30
                        row["peak_reserved_GiB"] = w.reserved / 2**30
                        row["device_used_GiB"] = device_used(dev) / 2**30
                    rows.append(row)
                    print("SPEED", json.dumps(row), flush=True)
        vae = None
        torch.cuda.empty_cache()
    return rows


def _child(args, extra: list[str]) -> list[dict]:
    cmd = [sys.executable, __file__, *sys.argv[1:], *extra]
    out = subprocess.run(cmd, capture_output=True, text=True)
    rows = [
        json.loads(line.split(" ", 1)[1])
        for line in out.stdout.splitlines()
        if line.startswith(("PREC ", "SPEED "))
    ]
    if out.returncode != 0:
        rows.append({"child": extra, "error": (out.stdout + out.stderr)[-1500:]})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtypes", default="fp32,bf16,fp16")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--speed-n", type=int, default=4)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--accels", default=",".join(ACCELS))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/bench/vae")
    args = ap.parse_args()
    dev = torch.device(args.device)
    dtypes = args.dtypes.split(",")
    if "bf16" in dtypes and not arch_of(dev).has_bf16:
        dtypes.remove("bf16")
        args.dtypes = ",".join(dtypes)
    match args.only.split(":"):
        case ["precision"]:
            precision(args)
            return
        case ["speed", name, accel]:
            speed(args, name, accel)
            return
    report = {
        "gpu": torch.cuda.get_device_name(dev),
        "args": vars(args),
        "precision": _child(args, ["--only", "precision"]),
        "speed": [],
    }
    for name in dtypes:
        for accel in args.accels.split(","):
            report["speed"] += _child(args, ["--only", f"speed:{name}:{accel}"])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{report['gpu'].replace(' ', '_')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
