"""Hardware-limit check of memory-knob combinations on a real-size slice.

For each combination (one fresh process each) it measures the step time, the PCIe bytes
the engines moved, and compares against the bounds: copy time = the largest of H2D bytes
/ pinned H2D bandwidth, D2H bytes / pinned D2H bandwidth and all bytes / measured
concurrent (both directions) bandwidth, compute time =
the same step with everything resident (measured), and the FLOP bound at the measured
MMA peak. `overlap` = step / max(copy, compute); 1.0 means fully overlapped. The
compute time of a GRAD_ACC case is the resident case at the same GRAD_ACC when it was
measured, else the GRAD_ACC=1 resident time scaled linearly.

    python scripts/bench/offload/limits.py --device cuda:0
Writes runs/bench/offload/limits_<gpu>.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from kohakumoss.offload.cpu_optim import CPUAdamW
from kohakumoss.training.memory.profiles import resolve
from kohakumoss.training.memory.session import MemorySession
from kohakumoss.training.memory.synthetic import SyntheticModel, synthetic_batch

MB = 1024 * 1024
CASES = {
    "full/resident/gpu": dict(stream="off", optim_state="gpu"),
    "full/resident/gpu/acc4": dict(stream="off", optim_state="gpu", acc=4),
    "full/grad_offload/gpu/acc4": dict(
        stream="off", optim_state="gpu", grad_offload=True, acc=4
    ),
    "full/stream_states": dict(stream="off", optim_state="stream", fused_step=True),
    "full/cpu_adamw": dict(stream="off", optim_state="cpu"),
    "full/swap/gpu_resident_states": dict(stream="swap", optim_state="gpu"),
    "full/swap/gpu_streamed_states": dict(stream="swap", optim_state="stream"),
    "full/swap/gpu_streamed_states/acc4": dict(
        stream="swap", optim_state="stream", acc=4
    ),
    "full/swap/cpu_adamw": dict(stream="swap", optim_state="cpu"),
    "adapter/resident": dict(stream="off", optim_state="gpu", full=False),
    "adapter/act_offload": dict(
        stream="off", optim_state="gpu", act_offload=True, full=False
    ),
    "adapter/ring": dict(stream="ring", optim_state="gpu", full=False),
    "adapter/ring/act_offload/grad_offload": dict(
        stream="ring",
        optim_state="gpu",
        act_offload=True,
        grad_offload=True,
        full=False,
    ),
    "adapter/ring/act_offload": dict(
        stream="ring", optim_state="gpu", act_offload=True, full=False
    ),
    "adapter/ring/grad_offload": dict(
        stream="ring", optim_state="gpu", grad_offload=True, full=False
    ),
    "adapter/ring/int8": dict(
        stream="ring", optim_state="gpu", base="int8", full=False
    ),
}


def bandwidth(device) -> dict[str, float]:
    n = 256 * MB // 2
    gpu = torch.empty(n, dtype=torch.float16, device=device)
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    out = {}
    for name, fn in (
        ("h2d", lambda: gpu.copy_(host, non_blocking=True)),
        ("d2h", lambda: host.copy_(gpu, non_blocking=True)),
    ):
        fn()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        torch.cuda.synchronize(device)
        out[name] = 5 * 256 * MB / (time.perf_counter() - t0) / 1e9
    gpu2 = torch.empty_like(gpu)
    host2 = torch.empty(n, dtype=torch.float16, pin_memory=True)
    s1, s2 = torch.cuda.Stream(device), torch.cuda.Stream(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.cuda.stream(s1):
            gpu.copy_(host, non_blocking=True)
        with torch.cuda.stream(s2):
            host2.copy_(gpu2, non_blocking=True)
    torch.cuda.synchronize(device)
    out["bidir_total"] = 2 * 5 * 256 * MB / (time.perf_counter() - t0) / 1e9
    return out


def mma_peak(device, dtype) -> float:
    a = torch.randn(8192, 8192, device=device, dtype=dtype)
    b = torch.randn(8192, 8192, device=device, dtype=dtype)
    a @ b
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(5):
        a @ b
    torch.cuda.synchronize(device)
    return 5 * 2 * 8192**3 / (time.perf_counter() - t0) / 1e12


def flops_per_step(args, full: bool, acc: int) -> float:
    d, f = args.dim, args.ffn
    per_token = 2 * (2 * d * d + 3 * f * d)
    passes = 1 + 1 + (2 if full else 1)
    return per_token * args.tokens * args.blocks * passes * acc


def run_case(args, name, device, dtype):
    spec = dict(CASES[name])
    acc = spec.pop("acc", 1)
    full = spec.pop("full", True)
    base = spec.pop("base", "compute")
    overrides = dict(checkpoint="block", stream_resident=2, base_precision=base, **spec)
    plan = resolve("balanced", overrides, device, trainable_base=full)
    torch.manual_seed(0)
    model = SyntheticModel(args.dim, args.ffn, args.blocks, 16).to(device, dtype)
    if not full:
        for p in model.layers.parameters():
            p.requires_grad_(False)
    session = MemorySession(
        model, plan, device, dtype, grad_acc=acc, trainable_base=full
    )
    side = int((args.tokens - args.text) ** 0.5)
    data = synthetic_batch([(side, side)], [args.text], 16, device, dtype)

    def step():
        for micro in range(acc):
            img, txt, sigma, seq, target = data
            loss = (
                (model(img, txt, sigma, seq).float() - target.float()).square().mean()
            )
            session.backward(loss, micro)
        session.step()

    engines = session.engines
    sources = [
        s
        for s in (
            engines.stream,
            engines.offload,
            session.grads,
            session.block_opt,
            session.dev_opt,
        )
        if s is not None and hasattr(s, "stats")
    ]
    step()
    torch.cuda.synchronize(device)

    def counters():
        h2d = sum(s.stats.get("bytes_h2d", 0) for s in sources)
        d2h = sum(s.stats.get("bytes_d2h", 0) for s in sources)
        if engines.offload is not None:
            h2d += engines.offload.stats["packed_bytes"]
            d2h += engines.offload.stats["packed_bytes"]
        return h2d, d2h

    h0, d0 = counters()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    h1, d1 = counters()
    h2d, d2h = (h1 - h0) / args.iters, (d1 - d0) / args.iters
    for opt in (session.dev_opt, session.host_block_opt):
        if isinstance(opt, CPUAdamW):
            n = sum(
                p.numel() * p.element_size()
                for g in opt.param_groups
                for p in g["params"]
                if p.device.type == "cuda"
            )
            h2d, d2h = h2d + n, d2h + n
    return {
        "case": name,
        "ms": 1e3 * sorted(times)[len(times) // 2],
        "peak_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
        "h2d_GB": h2d / 1e9,
        "d2h_GB": d2h / 1e9,
        "flops": flops_per_step(args, full, acc),
        "acc": acc,
        "full": full,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=4400)
    ap.add_argument("--text", type=int, default=300)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/bench/offload")
    args, _ = ap.parse_known_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    if args.only:
        print("ROW " + json.dumps(run_case(args, args.only, device, dtype)))
        return
    bw = bandwidth(device)
    peak = mma_peak(device, dtype)
    rows = []
    for name in args.cases.split(","):
        cmd = [sys.executable, __file__, "--only", name] + sys.argv[1:]
        out = subprocess.run(cmd, capture_output=True, text=True)
        found = [
            line[4:] for line in out.stdout.splitlines() if line.startswith("ROW ")
        ]
        if not found:
            rows.append({"case": name, "error": (out.stdout + out.stderr)[-600:]})
            print(json.dumps(rows[-1]))
            continue
        rows.append(json.loads(found[0]))
    resident = {
        (r["full"], r["acc"]): r
        for r in rows
        if r.get("case")
        in ("full/resident/gpu", "full/resident/gpu/acc4", "adapter/resident")
    }
    for r in rows:
        if "error" in r:
            continue
        copy_ms = 1e3 * max(
            r["h2d_GB"] / bw["h2d"],
            r["d2h_GB"] / bw["d2h"],
            (r["h2d_GB"] + r["d2h_GB"]) / bw["bidir_total"],
        )
        base = resident.get((r["full"], r["acc"])) or resident.get((r["full"], 1))
        compute_ms = base["ms"] * r["acc"] / base["acc"] if base else float("nan")
        r["copy_bound_ms"] = copy_ms
        r["compute_ms"] = compute_ms
        r["flop_bound_ms"] = 1e3 * r["flops"] / (peak * 1e12)
        r["overlap"] = r["ms"] / max(copy_ms, compute_ms)
        print(
            f"{r['case']:<42} ms={r['ms']:8.1f} copy={copy_ms:8.1f} compute={compute_ms:8.1f} "
            f"flop_bound={r['flop_bound_ms']:7.1f} overlap={r['overlap']:5.2f} "
            f"h2d={r['h2d_GB']:6.2f}GB d2h={r['d2h_GB']:6.2f}GB peak={r['peak_GiB']:5.2f}GiB"
        )
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "bandwidth_GBps": bw,
        "mma_TFLOPs": peak,
        "args": vars(args),
        "rows": rows,
    }
    print("bandwidth", bw, "mma TFLOP/s", round(peak, 1))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"limits_{report['gpu'].replace(' ', '_')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
