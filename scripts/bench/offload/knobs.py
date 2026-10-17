"""One measured line per VRAM knob, each in isolation, on MOSS-sized synthetic blocks,
built through MemorySession (the trainer's optimiser wiring); writes
runs/bench/offload/knobs_<gpu>.json. See docs/internals/offload.md.

    python scripts/bench/offload/knobs.py --device cuda:0
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from kohakumoss.offload.accel import default_type
from kohakumoss.offload.accel import module as accel
from kohakumoss.training.memory.profiles import resolve
from kohakumoss.training.memory.runners import build_runner
from kohakumoss.training.memory.session import MemorySession
from kohakumoss.training.memory.synthetic import SyntheticModel, synthetic_batch
from kohakumoss_kernels.hardware.arch import arch_of

GIB = 1024**3
FULL_FT = {
    "gpu_adamw": dict(optim_state="gpu"),
    "cpu_adamw": dict(optim_state="cpu"),
    "grad_offload_cpu_adamw": dict(optim_state="cpu", grad_offload=True),
    "grad_offload_gpu_adamw": dict(optim_state="gpu", grad_offload=True),
}
AGGRESSIVE = dict(stream="swap", stream_resident=2, grad_offload=True)


def model_and_data(args, device, dtype, tokens):
    torch.manual_seed(0)
    model = SyntheticModel(args.dim, args.ffn, args.blocks, 16).to(device, dtype)
    side = int((tokens - args.text) ** 0.5)
    data = synthetic_batch([(side, side)], [args.text], 16, device, dtype)
    return model, data


def loss_of(model, data):
    img, txt, sigma, seq, target = data
    return (model(img, txt, sigma, seq).float() - target.float()).square().mean()


def measure(name, device, step_fn, tokens, iters, stats=()):
    dm = accel(device)
    step_fn()
    dm.synchronize(device)
    dm.reset_peak_memory_stats(device)
    base = {k: s.stats.get(k, 0) for s in stats for k in ("bytes_h2d", "bytes_d2h")}
    times = []
    for _ in range(iters):
        dm.synchronize(device)
        t0 = time.perf_counter()
        step_fn()
        dm.synchronize(device)
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    row = {
        "case": name,
        "ms": med * 1e3,
        "ms_per_1k_tokens": med * 1e6 / tokens,
        "peak_GiB": dm.max_memory_allocated(device) / GIB,
        "resident_GiB": dm.memory_allocated(device) / GIB,
        "tokens": tokens,
    }
    for k in ("bytes_h2d", "bytes_d2h"):
        total = sum(s.stats.get(k, 0) for s in stats) - base.get(k, 0)
        row[k.replace("bytes_", "") + "_GB_per_step"] = total / 1e9 / iters
    return row


def _session_case(args, device, dtype, tokens, overrides, name):
    model, data = model_and_data(args, device, dtype, tokens)
    plan = resolve(
        "balanced",
        {"checkpoint": "block", **overrides},
        device,
        trainable_base=True,
    )
    session = MemorySession(
        model, plan, device, dtype, grad_acc=args.accum, trainable_base=True, lr=1e-5
    )
    stats = [
        s
        for s in (session.engines.stream, session.grads, session.block_opt)
        if s is not None and hasattr(s, "stats")
    ]

    def step():
        for micro in range(args.accum):
            session.backward(loss_of(model, data), micro)
        session.step()

    return measure(name, device, step, tokens, args.iters, stats)


def full_ft(args, device, dtype, mode):
    name = f"full_ft/{mode}/acc{args.accum}"
    return _session_case(args, device, dtype, args.tokens, FULL_FT[mode], name)


def granularity(args, device, dtype, value):
    model, data = model_and_data(args, device, dtype, args.tokens)
    for p in model.layers.parameters():
        p.requires_grad_(False)
    model.runner = build_runner(value, model.layers)

    def step():
        for _ in range(args.accum):
            loss_of(model, data).backward()
        model.zero_grad(set_to_none=True)

    return measure(
        f"checkpoint/{value}/acc{args.accum}", device, step, args.tokens, args.iters
    )


def aggressive(args, device, dtype, tokens):
    name = f"swap+grad_offload+stream_states/{tokens}tok/acc{args.accum}"
    overrides = {**AGGRESSIVE, "optim_state": "stream"}
    return _session_case(args, device, dtype, tokens, overrides, name)


CASES = {
    "full_ft:gpu_adamw": (full_ft, "gpu_adamw"),
    "full_ft:cpu_adamw": (full_ft, "cpu_adamw"),
    "full_ft:grad_offload_cpu_adamw": (full_ft, "grad_offload_cpu_adamw"),
    "full_ft:grad_offload_gpu_adamw": (full_ft, "grad_offload_gpu_adamw"),
    "ckpt:none": (granularity, "none"),
    "ckpt:block": (granularity, "block"),
    "ckpt:sub-block": (granularity, "sub-block"),
    "ckpt:selective": (granularity, "selective"),
    "ckpt:partial:0.5": (granularity, "partial:0.5"),
    "aggressive:tokens": (aggressive, "tokens"),
    "aggressive:big_tokens": (aggressive, "big_tokens"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=f"{default_type()}:0")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=4400)
    ap.add_argument("--big-tokens", type=int, default=17600)
    ap.add_argument("--text", type=int, default=300)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/bench/offload")
    args, rest = ap.parse_known_args()
    device = torch.device(args.device)
    dm = accel(device)
    dm.set_device(device)
    dtype = torch.bfloat16 if arch_of(device).has_bf16 else torch.float16
    if args.only:
        fn, arg = CASES[args.only]
        arg = {"tokens": args.tokens, "big_tokens": args.big_tokens}.get(arg, arg)
        print("ROW " + json.dumps(fn(args, device, dtype, arg)))
        return
    rows = []
    for label in args.cases.split(","):
        cmd = [sys.executable, __file__, "--only", label] + sys.argv[1:]
        out = subprocess.run(cmd, capture_output=True, text=True)
        found = [
            line[4:] for line in out.stdout.splitlines() if line.startswith("ROW ")
        ]
        rows.append(
            json.loads(found[0])
            if found
            else {"case": label, "error": (out.stdout + out.stderr)[-600:]}
        )
        print(json.dumps(rows[-1]))
    report = {
        "gpu": dm.get_device_name(device),
        "dtype": str(dtype),
        "args": vars(args),
        "rows": rows,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"knobs_{report['gpu'].replace(' ', '_')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
