"""One measured line per VRAM knob, each in isolation, on MOSS-sized synthetic blocks:
gradient offload + accumulation, checkpoint granularities under accumulation, and
streaming + gradient offload + checkpointing amortised over a larger token budget.

    python scripts/bench/offload/knobs.py --device cuda:0
Writes runs/bench/offload/knobs_<gpu>.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from kohakumoss.offload.cpu_optim import CPUAdamW
from kohakumoss.offload.grad import GradOffload
from kohakumoss.offload.stream import WeightStream
from kohakumoss.training.memory.runners import (
    CheckpointRunner,
    StreamRunner,
    build_runner,
)
from kohakumoss.training.memory.synthetic import SyntheticModel, synthetic_batch

GIB = 1024**3


def model_and_data(args, device, dtype, tokens):
    torch.manual_seed(0)
    model = SyntheticModel(args.dim, args.ffn, args.blocks, 16).to(device, dtype)
    side = int((tokens - args.text) ** 0.5)
    data = synthetic_batch([(side, side)], [args.text], 16, device, dtype)
    return model, data


def micro(model, data):
    img, txt, sigma, seq, target = data
    loss = (model(img, txt, sigma, seq).float() - target.float()).square().mean()
    loss.backward()


def measure(name, device, step_fn, tokens, iters, stats=()):
    step_fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = {k: s.stats.get(k, 0) for s in stats for k in ("bytes_h2d", "bytes_d2h")}
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step_fn()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    row = {
        "case": name,
        "ms": med * 1e3,
        "ms_per_1k_tokens": med * 1e6 / tokens,
        "peak_GiB": torch.cuda.max_memory_allocated(device) / GIB,
        "resident_GiB": torch.cuda.memory_allocated(device) / GIB,
        "tokens": tokens,
    }
    for k in ("bytes_h2d", "bytes_d2h"):
        total = sum(s.stats.get(k, 0) for s in stats) - base.get(k, 0)
        row[k.replace("bytes_", "") + "_GB_per_step"] = total / 1e9 / iters
    return row


def full_ft(args, device, dtype, mode):
    model, data = model_and_data(args, device, dtype, args.tokens)
    model.runner = CheckpointRunner("block").bind(model.layers)
    params = list(model.parameters())
    stats = []
    match mode:
        case "gpu_adamw":
            opt = torch.optim.AdamW(params, lr=1e-5, foreach=False)
            before = after = None
        case "cpu_adamw":
            opt = CPUAdamW(params, lr=1e-5)
            before = after = None
        case "grad_offload_cpu_adamw":
            off = GradOffload(params)
            opt = CPUAdamW(params, lr=1e-5)
            opt.use_host_grads(off)
            before, after, stats = off.flush, off.zero, [off]

    def step():
        for _ in range(args.accum):
            micro(model, data)
        if before is not None:
            before()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if after is not None:
            after()

    return measure(
        f"full_ft/{mode}/acc{args.accum}", device, step, args.tokens, args.iters, stats
    )


def granularity(args, device, dtype, value):
    model, data = model_and_data(args, device, dtype, args.tokens)
    for p in model.layers.parameters():
        p.requires_grad_(False)
    model.runner = build_runner(value, model.layers)

    def step():
        for _ in range(args.accum):
            micro(model, data)
        model.zero_grad(set_to_none=True)

    return measure(
        f"checkpoint/{value}/acc{args.accum}", device, step, args.tokens, args.iters
    )


def aggressive(args, device, dtype, tokens):
    model, data = model_and_data(args, device, dtype, tokens)
    engine = WeightStream(model.layers, device, resident=2, trainable=True, dtype=dtype)
    model.runner = StreamRunner(CheckpointRunner("block"), engine).bind(model.layers)
    resident = [p for p in model.parameters() if p.device.type == "cuda"]
    off = GradOffload(resident)
    host_opt = CPUAdamW(list(engine.params.values()), lr=1e-5)
    dev_opt = CPUAdamW(resident, lr=1e-5)
    dev_opt.use_host_grads(off)

    def step():
        for _ in range(args.accum):
            micro(model, data)
        engine.flush_grads()
        off.flush()
        host_opt.step()
        dev_opt.step()
        host_opt.zero_grad(set_to_none=True)
        off.zero()
        engine.refresh()

    return measure(
        f"stream+grad_offload+ckpt/acc{args.accum}",
        device,
        step,
        tokens,
        args.iters,
        [engine, off],
    )


CASES = {
    "full_ft:gpu_adamw": (full_ft, "gpu_adamw"),
    "full_ft:cpu_adamw": (full_ft, "cpu_adamw"),
    "full_ft:grad_offload_cpu_adamw": (full_ft, "grad_offload_cpu_adamw"),
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
    ap.add_argument("--device", default="cuda:0")
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
    if args.only:
        torch.cuda.set_device(device)
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        fn, arg = CASES[args.only]
        arg = {"tokens": args.tokens, "big_tokens": args.big_tokens}.get(arg, arg)
        print("ROW " + json.dumps(fn(args, device, dtype, arg)))
        return
    rows = []
    for label in args.cases.split(","):
        cmd = [sys.executable, __file__, "--only", label] + sys.argv[1:]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        found = [line[4:] for line in out.splitlines() if line.startswith("ROW ")]
        rows.append(
            json.loads(found[0]) if found else {"case": label, "error": out[-400:]}
        )
        print(json.dumps(rows[-1]))
    dtype = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    report = {
        "gpu": torch.cuda.get_device_name(device),
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
