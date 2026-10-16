"""Offload bench: copy bandwidth (pinned / pageable / staged) and the step cost of weight
streaming and activation offload on MOSS-sized synthetic blocks.

    python scripts/bench/offload/stream.py --device cuda:0 --blocks 8 --tokens 4400
Writes runs/bench/offload/<gpu>.json.
"""

import argparse
import json
import os
import time

import torch

from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.offload.stream import WeightStream
from kohakumoss.training.memory.runners import (
    CheckpointRunner,
    OffloadRunner,
    PlainRunner,
    StreamRunner,
)
from kohakumoss.training.memory.synthetic import SyntheticModel, synthetic_batch

MB = 1024 * 1024


def bandwidth(device: torch.device, sizes_mb=(4, 64, 256)) -> list[dict]:
    rows = []
    for size in sizes_mb:
        n = size * MB // 2
        gpu = torch.empty(n, dtype=torch.float16, device=device)
        for kind in ("pinned", "pageable"):
            host = torch.empty(n, dtype=torch.float16, pin_memory=kind == "pinned")
            for direction in ("h2d", "d2h"):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                for _ in range(5):
                    if direction == "h2d":
                        gpu.copy_(host, non_blocking=True)
                    else:
                        host.copy_(gpu, non_blocking=True)
                torch.cuda.synchronize(device)
                dt = (time.perf_counter() - t0) / 5
                rows.append(
                    {
                        "size_mb": size,
                        "host": kind,
                        "dir": direction,
                        "GBps": size / 1024 / dt,
                    }
                )
    return rows


def step_case(name, device, args, dtype):
    torch.manual_seed(0)
    model = SyntheticModel(args.dim, args.ffn, args.blocks, 16).to(device, dtype)
    for p in model.layers.parameters():
        p.requires_grad_(False)
    side = int((args.tokens - args.text) ** 0.5)
    data = synthetic_batch([(side, side)], [args.text], 16, device, dtype)
    engine = offload = None
    match name:
        case "resident":
            model.runner = PlainRunner()
        case "checkpoint":
            model.runner = CheckpointRunner("block")
        case "act_offload":
            offload = ActivationOffload(device)
            offload.exclude(model.parameters())
            model.runner = OffloadRunner(CheckpointRunner("block"), offload)
        case "stream_pinned" | "stream_staged":
            engine = WeightStream(
                model.layers, device, resident=2, pinned=name == "stream_pinned"
            )
            model.runner = StreamRunner(CheckpointRunner("block"), engine)
    model.runner.bind(model.layers)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(args.iters + 1):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        img, txt, sigma, seq, target = data
        loss = (model(img, txt, sigma, seq).float() - target.float()).square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        if i:
            times.append(time.perf_counter() - t0)
        model.zero_grad(set_to_none=True)
    row = {
        "case": name,
        "ms": 1e3 * sorted(times)[len(times) // 2],
        "peak_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
        "tokens_per_s": args.tokens / (sorted(times)[len(times) // 2]),
    }
    if engine is not None:
        row["h2d_GB_per_step"] = engine.stats["bytes_h2d"] / 1e9 / (args.iters + 1)
    if offload is not None:
        row["offloaded_GB_per_step"] = (
            offload.stats["packed_bytes"] / 1e9 / (args.iters + 1)
        )
    del model, engine, offload
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=4400)
    ap.add_argument("--text", type=int, default=300)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default="runs/bench/offload")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    major, _ = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "args": vars(args),
        "bandwidth": bandwidth(device),
        "steps": [
            step_case(n, device, args, dtype)
            for n in (
                "resident",
                "checkpoint",
                "act_offload",
                "stream_pinned",
                "stream_staged",
            )
        ],
    }
    os.makedirs(args.out, exist_ok=True)
    name = report["gpu"].replace(" ", "_").replace("/", "_")
    path = os.path.join(args.out, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for row in report["bandwidth"]:
        print(
            f"bw {row['size_mb']:>4}MB {row['host']:>8} {row['dir']} {row['GBps']:6.2f} GB/s"
        )
    for row in report["steps"]:
        print(
            " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            )
        )
    print("wrote", path)


if __name__ == "__main__":
    main()
