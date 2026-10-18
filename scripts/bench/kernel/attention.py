"""Varlen attention at MOSS-Image shapes: every backend vs the PyTorch paths, speed,
memory and error against an fp64 eager reference.

    python scripts/bench/kernel/attention.py --device cuda --dtype fp16 --out runs/bench/attention

See docs/performance/attention.md.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakumoss_kernels as kk
from kohakumoss_kernels import compile_setup
from kohakumoss_kernels.backends.eager.attention import attention_varlen as sdpa_loop
from kohakumoss_kernels.backends.varlen_choice import VarlenChoice
from kohakumoss_kernels.error_stats import stats
from kohakumoss_kernels.hardware.device import measure_rates
from kohakumoss_kernels.hardware.memory import device_used, window
from kohakumoss_kernels.tuning.timing import best_ms

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None
try:
    from kohakumoss_kernels.backends.triton.attention import varlen as triton_kernels
except ImportError:
    triton_kernels = None
try:
    from kohakumoss_kernels.backends.tilelang.attention import (
        varlen as tilelang_kernels,
    )
except ImportError:
    tilelang_kernels = None
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    flex_attention = None

CASES = {
    "joint-1MP-x2": [4096 + 300] * 2,
    "joint-2MP-x1": [8192 + 500],
    "joint-pack-mixed": [4096 + 300, 2304 + 120, 1024 + 600],
    "joint-512px-x8": [1024 + 80] * 8,
    "text-refiner": [77, 300, 512, 150, 600, 40],
}
HEADS, HEAD_DIM = 24, 128
REF_HEADS = slice(0, 2)
KERNELS = (("tilelang", tilelang_kernels), ("triton", triton_kernels))


def padded_sdpa(q, k, v, cu, max_len, scale):
    """Gather segments into [B, L] with a key mask, one SDPA call, scatter back."""
    lens = (cu[1:] - cu[:-1]).tolist()
    b, (_, h, d) = len(lens), q.shape
    idx = torch.cat(
        [
            torch.arange(s, s + ln, device=q.device)
            for s, ln in zip(cu[:-1].tolist(), lens)
        ]
    )
    row = torch.cat(
        [torch.full((ln,), i, device=q.device) for i, ln in enumerate(lens)]
    )
    col = idx - cu[:-1].long()[row]
    pads = []
    for t in (q, k, v):
        p = t.new_zeros(b, max_len, h, d)
        p[row, col] = t[idx]
        pads.append(p.transpose(1, 2))
    mask = (
        torch.arange(max_len, device=q.device)[None, :]
        < torch.tensor(lens, device=q.device)[:, None]
    )
    o = F.scaled_dot_product_attention(
        *pads, attn_mask=mask[:, None, None, :], scale=scale
    )
    return o.transpose(1, 2)[row, col]


def torch_varlen(q, k, v, cu, max_len, scale):
    return varlen_attn(q, k, v, cu, cu, max_len, max_len, scale=scale)


def make_flex(cu, n):
    """flex_attention (compiled) with a document mask over the segments of `cu`."""
    seg = torch.repeat_interleave(
        torch.arange(cu.numel() - 1, device=cu.device), (cu[1:] - cu[:-1]).long()
    )

    def doc(b, h, qi, ki):
        return seg[qi] == seg[ki]

    mask = create_block_mask(doc, None, None, n, n, device=cu.device)
    compile_setup.apply()
    fn = torch.compile(flex_attention)

    def run(q, k, v, cu, max_len, scale):
        o = fn(
            *(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)),
            block_mask=mask,
            scale=scale,
        )
        return o.squeeze(0).transpose(0, 1)

    return run


def _sm80(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def implementations(device, dtype, cu, n):
    """{name: fn}: the PyTorch paths, each backend's registry entry (`_chosen`, its
    per-shape choice) and its kernels alone (`_kernels`)."""
    compile_setup.apply()
    impls = {
        "sdpa_loop": sdpa_loop,
        "sdpa_loop_compiled": torch.compile(sdpa_loop, dynamic=True),
        "sdpa_padded": padded_sdpa,
    }
    if varlen_attn is not None and _sm80(device):
        impls["torch_varlen_attn"] = torch_varlen
    if flex_attention is not None and (_sm80(device) or device.type == "xpu"):
        impls["flex_doc_mask"] = make_flex(cu, n)
    for backend, kernels in KERNELS:
        try:
            impls[f"kohakumoss_{backend}_chosen"] = kk.resolve(
                "attention_varlen", device, dtype, backend=backend
            )
        except kk.registry.NoImplementationError:
            continue
        if kernels is not None:
            impls[f"kohakumoss_{backend}_kernels"] = kernels.attention_varlen
    return impls


def bound_impl(fn, q, segments: int, max_len: int) -> str | None:
    """Name of the implementation a per-shape choice bound for this problem (None for
    a plain implementation, "undecided" before the choice settled)."""
    choice = getattr(fn, "fn", fn)
    if not isinstance(choice, VarlenChoice):
        return None
    bound = choice.bound.get((q.device, choice.key(q, segments, max_len)))
    names = {id(f): name for name, f in choice.impls(q.device).items()}
    return "undecided" if bound is None else names.get(id(bound), "?")


def rel(a, r):
    """Relative Frobenius error of `a` against the fp64 reference `r`."""
    return (
        (a.to(r.device, torch.float64) - r).norm() / r.norm().clamp_min(1e-30)
    ).item()


def interleaved(steps, device, iters, rounds):
    """{impl: [ms per round]}: every round times each step once (best of `iters`)."""
    times = {impl: [] for impl in steps}
    for _ in range(rounds):
        for impl, step in steps.items():
            times[impl].append(best_ms(step, device, iters))
    return times


def run_case(name, lens, device, mem, dtype, iters, only, rounds=1):
    """Rows of one case; `mem` is the device module (torch.cuda / torch.xpu). fwd + bwd
    is the median over `rounds` interleaved rounds (min in `fwd_bwd_min_ms`)."""
    n = sum(lens)
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    max_len, scale = max(lens), HEAD_DIM**-0.5
    q, k, v, do = (
        torch.randn(n, HEADS, HEAD_DIM, device=device, dtype=dtype) for _ in range(4)
    )
    # fp64 reference on the device where it is native (CUDA), else on the CPU
    ref_dev = device if device.type == "cuda" else torch.device("cpu")
    ref_in = [
        t[:, REF_HEADS].to(ref_dev, torch.float64).requires_grad_() for t in (q, k, v)
    ]
    ref = sdpa_loop(*ref_in, cu.to(ref_dev), max_len, scale)
    ref.backward(do[:, REF_HEADS].to(ref_dev, torch.float64))
    rows, steps = [], {}
    for impl, fn in implementations(device, dtype, cu, n).items():
        if only and not any(s in impl for s in only):
            continue
        leaves = [t.clone().requires_grad_() for t in (q, k, v)]
        try:
            out = fn(*leaves, cu, max_len, scale)
            out.backward(do)
        except Exception as err:  # noqa: BLE001
            rows.append(
                {
                    "case": name,
                    "impl": impl,
                    "error": f"{type(err).__name__}: {err}"[:200],
                }
            )
            continue
        got = {"o": out[:, REF_HEADS]} | {
            f"d{c}": t.grad[:, REF_HEADS] for c, t in zip("qkv", leaves)
        }
        refs = {"o": ref} | {f"d{c}": r.grad for c, r in zip("qkv", ref_in)}
        errs = {key: rel(got[key], refs[key]) for key in got}
        error_stats = {key: stats(got[key].to(ref_dev), refs[key]) for key in got}
        del out, got
        fwd_in = [t.detach() for t in (q, k, v)]
        with torch.no_grad():
            fwd = best_ms(lambda: fn(*fwd_in, cu, max_len, scale), device, iters)

        def step(fn=fn, leaves=leaves):
            for t in leaves:
                t.grad = None
            fn(*leaves, cu, max_len, scale).backward(do)

        base = mem.memory_allocated(device)
        with window(device) as w:
            step()
        peak = (w.alloc - base) / 2**20
        steps[impl] = step
        rows.append(
            {
                "case": name,
                "impl": impl,
                "bound": bound_impl(fn, q, len(lens), max_len),
                "fwd_ms": fwd,
                "peak_mib": peak,
                "peak_reserved_mib": w.reserved / 2**20,
                "device_used_mib": device_used(device) / 2**20,
                **errs,
                "stats": error_stats,
            }
        )
    times = interleaved(steps, device, iters, rounds)
    for row in rows:
        if row["impl"] in times:
            row["fwd_bwd_ms"] = statistics.median(times[row["impl"]])
            row["fwd_bwd_min_ms"] = min(times[row["impl"]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=1, help="interleaved fwd+bwd rounds")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--out", default="runs/bench/attention")
    ap.add_argument("--impls", nargs="*", default=[], help="substring filter")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:1, xpu, xpu:0, ...")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    mem = getattr(torch, device.type)
    if device.index is None:
        device = torch.device(device.type, mem.current_device())
    mem.set_device(device)
    dev = measure_rates(device)
    rows = []
    for name in args.cases:
        rows += run_case(
            name, CASES[name], device, mem, dtype, args.iters, args.impls, args.rounds
        )
        mem.empty_cache()
    report = {
        "device": dev.key,
        "mma_tflops": dev.mma_tflops,
        "dram_gbs": dev.dram_gbs,
        "dtype": args.dtype,
        "iters": args.iters,
        "rounds": args.rounds,
        "rows": rows,
    }
    tag = "-".join(args.impls) or "all"
    out = Path(args.out) / f"{dev.key}-{args.dtype}-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{dev.key} {args.dtype}")
    for r in rows:
        if "error" in r:
            print(f"{r['case']:18s} {r['impl']:26s} ERROR {r['error']}")
            continue
        bound = f" -> {r['bound']}" if r["bound"] else ""
        print(
            f"{r['case']:18s} {r['impl']:26s} fwd {r['fwd_ms']:8.3f}  fwd+bwd {r['fwd_bwd_ms']:8.3f}  "
            f"peak {r['peak_mib']:7.0f}MiB  err o {r['o']:.2e} dq {r['dq']:.2e} dk {r['dk']:.2e} dv {r['dv']:.2e}"
            f"{bound}"
        )


if __name__ == "__main__":
    main()
