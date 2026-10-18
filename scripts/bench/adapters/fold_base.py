"""Bench the LoRA bypass with the base add folded into the LyCORIS kernel's store against
the separate add (base + delta), forward and forward + backward, at a MOSS layer shape.

    python scripts/bench/adapters/fold_base.py --backend triton
"""

import argparse
import json

import torch
import torch.nn.functional as F
from lycoris.kernels.autograd import locon_bypass_diff


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="triton")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=3072)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    p.add_argument("--out", default=None)
    return p.parse_args()


def separate(x, w, d, u, gamma, backend):
    base = F.linear(x, w)
    return base + locon_bypass_diff(x, d.to(x.dtype), u.to(x.dtype), gamma, backend)


def folded(x, w, d, u, gamma, backend):
    base = F.linear(x, w)
    dd, uu = d.to(x.dtype), u.to(x.dtype)
    return locon_bypass_diff(x, dd, uu, gamma, backend, base=base)


def timed(fn, args, backward: bool, iters: int) -> float:
    """Median milliseconds per call (forward, or forward + backward)."""
    times = []
    for i in range(iters + 10):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        y = fn(*args)
        if backward:
            y.backward(torch.ones_like(y))
        end.record()
        torch.cuda.synchronize()
        if i >= 10:
            times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def main():
    args = parse()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(args.tokens, args.dim, device=dev, dtype=dtype)
    w = (torch.randn(args.dim, args.dim, device=dev) / 55).to(dtype)
    d = (torch.randn(args.rank, args.dim, device=dev) / 55).requires_grad_(True)
    u = (torch.randn(args.dim, args.rank, device=dev) / 10).requires_grad_(True)
    rows = {}
    for backward in (False, True):
        x.requires_grad_(backward)
        call = (x, w, d, u, 0.5, args.backend)
        key = "fwd+bwd" if backward else "fwd"
        with torch.set_grad_enabled(backward):
            a = timed(separate, call, backward, args.iters)
            b = timed(folded, call, backward, args.iters)
        rows[key] = {"separate_ms": a, "folded_ms": b, "speedup": a / b}
        print(f"{key:8s} separate {a:.4f} ms  folded {b:.4f} ms  x{a / b:.3f}")
    meta = vars(args) | {"device": torch.cuda.get_device_name()}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
