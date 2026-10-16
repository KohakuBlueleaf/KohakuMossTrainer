"""Step bench of the real MOSS architecture (random weights) per memory profile:
time, peak VRAM, tokens/s, and the memory model's estimate next to the measurement.

    python scripts/bench/step/step.py --device cuda:0 --profiles balanced,low,minimal
Writes runs/bench/step/<gpu>_<arch>.json. Trainable parameters are the per-block norm
weights plus the embedders (adapter-like gradient flow through every block).
"""

import argparse
import gc
import json
import os
import time

import torch

from kohakumoss.model import build_backbone, resolve_config
from kohakumoss.training.memory.apply import apply_to_backbone
from kohakumoss.training.memory.estimate import Workload, estimate
from kohakumoss.training.memory.profiles import resolve
from kohakumoss.types.seqinfo import build_seqinfo


def batch(cfg, tokens: int, text: int, device, dtype):
    side = int((tokens - text) ** 0.5)
    seq = build_seqinfo(torch.tensor([[side, side]]), torch.tensor([text])).to(device)
    img = torch.randn(seq.n_img, cfg.latent_channels, device=device, dtype=dtype)
    txt = torch.randn(seq.n_txt, cfg.text_dim, device=device, dtype=dtype)
    sigma = torch.tensor([0.6], device=device)
    return img, txt, sigma, seq


def run_profile(name, args, device, dtype, cfg):
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    build_device = "cpu" if name == "minimal" else device
    model = build_backbone(cfg, device=build_device, dtype=dtype, init=True)
    trainable = 0
    for n, p in model.named_parameters():
        keep = "norm" in n or n.startswith("video_embedder")
        p.requires_grad_(keep)
        trainable += p.numel() if keep else 0
    plan = resolve(name, args.overrides and json.loads(args.overrides), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    apply_to_backbone(model, plan, device, dtype, training=True)
    model.bind_ops()
    apply_peak = torch.cuda.max_memory_allocated(device)
    resident = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    data = batch(cfg, args.tokens, args.text, device, dtype)
    times = []
    for i in range(args.iters + 1):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        v = model(*data)
        v.float().square().mean().backward()
        torch.cuda.synchronize(device)
        if i:
            times.append(time.perf_counter() - t0)
        model.zero_grad(set_to_none=True)
    med = sorted(times)[len(times) // 2]
    work = Workload(cfg, args.tokens - args.text, args.text, trainable, compute_bytes=2)
    est = estimate(work, plan)
    row = {
        "profile": name,
        "plan": {k: v for k, v in plan.__dict__.items() if k != "kernels"},
        "ms": med * 1e3,
        "tokens_per_s": args.tokens / med,
        "peak_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
        "resident_GiB": resident / 1024**3,
        "apply_peak_GiB": apply_peak / 1024**3,
        "baseline_GiB": baseline / 1024**3,
        "estimate_GiB": est.gib(),
    }
    del model
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--profiles", default="balanced,low,minimal")
    ap.add_argument("--overrides", default="")
    ap.add_argument("--tokens", type=int, default=4400)
    ap.add_argument("--text", type=int, default=300)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="runs/bench/step")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    major, _ = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    cfg = resolve_config(args.variant)
    rows = []
    for name in args.profiles.split(","):
        try:
            rows.append(run_profile(name, args, device, dtype, cfg))
        except torch.OutOfMemoryError as err:
            rows.append({"profile": name, "error": "OOM", "detail": str(err)[:200]})
        print(json.dumps(rows[-1]))
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "args": vars(args),
        "rows": rows,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(
        args.out, f"{report['gpu'].replace(' ', '_')}_{args.variant}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
