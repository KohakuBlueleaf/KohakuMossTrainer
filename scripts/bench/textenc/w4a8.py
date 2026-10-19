"""The two decode paths of a W4A8 text-encoder layer, per Qwen3.5 linear shape.

Per shape (N, K) and token count M, in fp16 and bf16 (where the card has it): the
codes decoded once to int8 (`W4A8Weight.to_int8`, then the rotated int8 linear) against
the packed codes decoded per call (`W4A8Weight.linear`: the compiled decode, then the
same linear), the decode alone (eager and compiled), and PyTorch's dense linear on the
dequantized weight; device ms (profiler kernel time), the error against fp64 and the
bytes per layer of each form. One JSON per card.

    python scripts/bench/textenc/w4a8.py --out runs/textenc --device cuda
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from kohakumoss.quant.w4a8 import W4A8Weight, decode_codes, decoder_for
from kohakumoss_kernels.hadamard import hadamard_matrix
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.tuning.timing import graph_ms

SHAPES = {
    "q_proj": (8192, 4096),
    "kv_proj": (1024, 4096),
    "o_proj": (4096, 4096),
    "in_proj_qkv": (8192, 4096),
    "gate_up": (12288, 4096),
    "down": (4096, 12288),
}
TOKENS = (512, 2048, 8192)
GROUP, ROTATION = 16, 256
LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210,
    -0.050720, 0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876,
    0.980671,
)  # fmt: skip
PROFILED = {
    "cuda": (torch.profiler.ProfilerActivity.CUDA, torch.autograd.DeviceType.CUDA),
    "xpu": (torch.profiler.ProfilerActivity.XPU, torch.autograd.DeviceType.XPU),
}


def kernel_ms(device: torch.device, fn, iters: int = 5) -> float:
    """Device time per call (profiler sum of kernels, host dispatch excluded)."""
    module = getattr(torch, device.type)
    activity, kind = PROFILED[device.type]
    for _ in range(2):
        fn()
    module.synchronize(device)
    with torch.profiler.profile(activities=[activity]) as prof:
        for _ in range(iters):
            fn()
        module.synchronize(device)
    us = sum(e.device_time_total for e in prof.key_averages() if e.device_type == kind)
    return us / iters / 1e3 if us > 0 else graph_ms(fn, device)


def layer(n: int, k: int, device) -> W4A8Weight:
    """A W4A8 weight of random 4-bit codes, e4m3 group scales and row scales."""
    g = torch.Generator(device="cpu").manual_seed(n * 7 + k)
    q = torch.randint(-128, 128, (n, k // 2), generator=g, dtype=torch.int8)
    s_rel = (torch.rand(n, k // GROUP, generator=g) * 120 + 8).to(torch.float8_e4m3fn)
    s_channel = torch.rand(n, generator=g) * 1e-3 + 1e-4
    w = W4A8Weight(q, s_rel, s_channel, torch.tensor(LUT), GROUP, ROTATION)
    return w.to(device)


def rel_err(got, ref) -> float:
    return ((got.double() - ref).abs().max() / ref.abs().max()).item()


def run_case(device, dtype, name, n, k, m) -> dict:
    packed = layer(n, k, device)
    packed.bind(dtype, device)
    int8 = packed.to_int8()
    int8.bind(dtype, device)
    x = torch.randn(m, k, device=device, dtype=dtype)
    h = hadamard_matrix(ROTATION, torch.float32, device).double()
    w64 = int8.q.double() * int8.scale.double()[:, None]
    ref = (x.double().reshape(-1, ROTATION) @ h).reshape(x.shape) @ w64.T
    dense = int8.weight(dtype)
    compiled = decoder_for(device)
    parts = (packed.q, packed.s_rel, packed.codebook, GROUP, packed.bits)
    paths = {
        "int8": lambda: int8.linear(x, None),
        "packed": lambda: packed.linear(x, None),
        "dense_pytorch": lambda: F.linear(x, dense),
    }
    out = {}
    with torch.no_grad():
        for path, fn in paths.items():
            out[path] = {"ms": kernel_ms(device, fn), "rel_err": rel_err(fn(), ref)}
        out["decode_eager"] = {"ms": kernel_ms(device, lambda: decode_codes(*parts))}
        out["decode_compiled"] = {"ms": kernel_ms(device, lambda: compiled(*parts))}
    row = {
        "dtype": str(dtype),
        "case": name,
        "n": n,
        "k": k,
        "m": m,
        "paths": out,
        "bytes": {
            "packed": packed.nbytes(),
            "int8": int8.nbytes(),
            "dense": dense.nbytes,
        },
        "packed_over_int8": out["packed"]["ms"] / out["int8"]["ms"],
    }
    print(
        f"{str(dtype)[6:]:8s} {name:12s} m={m:6d} "
        + " ".join(f"{p}={v['ms']:.3f}" for p, v in out.items())
        + f" | packed/int8 x{row['packed_over_int8']:.3f}",
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/textenc")
    ap.add_argument("--device", default="cuda", help="cuda[:i] or xpu[:i]")
    ap.add_argument("--tokens", default="", help="comma list of token counts")
    args = ap.parse_args()
    torch._dynamo.config.recompile_limit = 512
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device(device.type, 0)
    getattr(torch, device.type).set_device(device.index)
    arch = arch_of(device)
    dtypes = [torch.float16] + ([torch.bfloat16] if arch.has_bf16 else [])
    tokens = tuple(int(t) for t in args.tokens.split(",")) if args.tokens else TOKENS
    rows = [
        run_case(device, dtype, name, n, k, m)
        for dtype in dtypes
        for name, (n, k) in SHAPES.items()
        for m in tokens
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    card = getattr(torch, device.type).get_device_name(device.index).replace(" ", "_")
    with open(out / f"w4a8_{card}.json", "w", encoding="utf-8") as fh:
        json.dump({"device": card, "arch": arch.name, "rows": rows}, fh, indent=1)
    print(f"wrote {out / f'w4a8_{card}.json'}")


if __name__ == "__main__":
    main()
