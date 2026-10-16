"""Varlen attention: every backend vs PyTorch at MOSS-Image shapes, speed and accuracy.

    python scripts/bench/kernel/attention.py --dtype fp16 --out runs/bench/attention

Rows: case x implementation, forward and forward+backward device ms (best of N, CUDA
events), peak memory, and relative error vs the fp64 eager reference for o, dq, dk, dv.
PyTorch baselines: per-segment SDPA loop (eager and torch.compile), padded SDPA with a
key mask, torch varlen_attn and flex_attention with a document mask where available.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakumoss_kernels as kk
from kohakumoss_kernels.backends.eager.attention import attention_varlen as sdpa_loop
from kohakumoss_kernels.hardware.device import probe
from kohakumoss_kernels.tuning.timing import best_ms

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None
try:
    from kohakumoss_kernels.backends.triton import attention as triton_kernels
except ImportError:
    triton_kernels = None
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
    seg = torch.repeat_interleave(
        torch.arange(cu.numel() - 1, device=cu.device), (cu[1:] - cu[:-1]).long()
    )

    def doc(b, h, qi, ki):
        return seg[qi] == seg[ki]

    mask = create_block_mask(doc, None, None, n, n, device=cu.device)
    fn = torch.compile(flex_attention)

    def run(q, k, v, cu, max_len, scale):
        o = fn(
            *(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)),
            block_mask=mask,
            scale=scale,
        )
        return o.squeeze(0).transpose(0, 1)

    return run


def implementations(device, dtype, cu, n):
    impls = {
        "sdpa_loop": sdpa_loop,
        "sdpa_loop_compiled": torch.compile(sdpa_loop, dynamic=True),
        "sdpa_padded": padded_sdpa,
    }
    if varlen_attn is not None and torch.cuda.get_device_capability(device)[0] >= 8:
        impls["torch_varlen_attn"] = torch_varlen
    if flex_attention is not None and torch.cuda.get_device_capability(device)[0] >= 8:
        impls["flex_doc_mask"] = make_flex(cu, n)
    for backend in ("tilelang", "triton"):
        try:
            impls[f"kohakumoss_{backend}"] = kk.resolve(
                "attention_varlen", device, dtype, backend=backend
            )
        except kk.registry.NoImplementationError:
            continue
    if "kohakumoss_triton" in impls:
        impls["kohakumoss_triton_raced"] = impls.pop("kohakumoss_triton")
        impls["kohakumoss_triton"] = triton_kernels.attention_varlen
    return impls


def rel(a, r):
    return ((a.double() - r).norm() / r.norm().clamp_min(1e-30)).item()


def run_case(name, lens, device, dtype, iters, only):
    n = sum(lens)
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    max_len, scale = max(lens), HEAD_DIM**-0.5
    q, k, v, do = (
        torch.randn(n, HEADS, HEAD_DIM, device=device, dtype=dtype) for _ in range(4)
    )
    ref_in = [t[:, REF_HEADS].double().requires_grad_() for t in (q, k, v)]
    ref = sdpa_loop(*ref_in, cu, max_len, scale)
    ref.backward(do[:, REF_HEADS].double())
    rows = []
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
        errs = {"o": rel(out[:, REF_HEADS], ref)} | {
            f"d{c}": rel(t.grad[:, REF_HEADS], r.grad)
            for c, t, r in zip("qkv", leaves, ref_in)
        }
        del out
        fwd_in = [t.detach() for t in (q, k, v)]
        with torch.no_grad():
            fwd = best_ms(
                lambda: fn(*fwd_in, cu, max_len, scale), torch.device(device), iters
            )

        def step():
            for t in leaves:
                t.grad = None
            fn(*leaves, cu, max_len, scale).backward(do)

        torch.cuda.reset_peak_memory_stats(device)
        both = best_ms(step, torch.device(device), iters)
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        rows.append(
            {
                "case": name,
                "impl": impl,
                "fwd_ms": fwd,
                "fwd_bwd_ms": both,
                "peak_mib": peak,
                **errs,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--out", default="runs/bench/attention")
    ap.add_argument("--impls", nargs="*", default=[], help="substring filter")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda"
    dev = probe(device)
    rows = []
    for name in args.cases:
        rows += run_case(name, CASES[name], device, dtype, args.iters, args.impls)
        torch.cuda.empty_cache()
    report = {
        "device": dev.key,
        "mma_tflops": dev.mma_tflops,
        "dram_gbs": dev.dram_gbs,
        "dtype": args.dtype,
        "rows": rows,
    }
    tag = "-".join(args.impls) or "all"
    out = Path(args.out) / f"{dev.key}-{args.dtype}-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"{dev.key} {args.dtype}")
    for r in rows:
        if "error" in r:
            print(f"{r['case']:18s} {r['impl']:22s} ERROR {r['error']}")
            continue
        print(
            f"{r['case']:18s} {r['impl']:22s} fwd {r['fwd_ms']:8.3f}  fwd+bwd {r['fwd_bwd_ms']:8.3f}  "
            f"peak {r['peak_mib']:7.0f}MiB  err o {r['o']:.2e} dq {r['dq']:.2e} dk {r['dk']:.2e} dv {r['dv']:.2e}"
        )


if __name__ == "__main__":
    main()
