"""Quantized frozen-base linear vs PyTorch: fused kernels against dequantize+matmul.

Arms per case: `eager` (dequantize to the activation dtype, then F.linear),
`compile` (torch.compile of the same), and every registered quant_linear backend
pinned. A backend arm runs what its tuner selected for the shape -- one of its
kernel configs or a PyTorch path -- so its row says what training gets. A
backend that cannot serve the device / dtype / format gets a row carrying the
registry's reason. Rows carry kernel time (profiler, host dispatch excluded) for
forward and fwd+bwd(dx), the error against fp64, and `race`: the tuner's own
device times of its best kernel config against the eager and compile paths,
for the forward and the dgrad GEMM separately. The verdict says whether the
best kernel beats both PyTorch paths (`kernel`) and whether the selected path
is within 3% of the faster PyTorch path (`selected`). One JSON per card.

    python scripts/bench/kernel/quant.py --out runs/bench/quant --device xpu
"""

import argparse
import json
from pathlib import Path

import torch

from kohakumoss_kernels import registry, tuning
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.quant import quant_linear_reference
from kohakumoss_kernels.tuning.timing import graph_ms

SHAPES = {
    "q": (3072, 3072),
    "qkv": (9216, 3072),
    "up_gate": (16384, 3072),
    "down": (3072, 8192),
}
TOKENS = (300, 4096, 16384)
TABLES = ("quant_fwd", "quant_dgrad", "quant_fwd_tl", "quant_dgrad_tl")
PROFILED = {
    "cuda": (torch.profiler.ProfilerActivity.CUDA, torch.autograd.DeviceType.CUDA),
    "xpu": (torch.profiler.ProfilerActivity.XPU, torch.autograd.DeviceType.XPU),
}


class Device:
    """The bench device, its torch device module and profiler activity, resolved once."""

    def __init__(self, spec: str):
        self.device = torch.device(spec)
        if self.device.index is None:
            self.device = torch.device(self.device.type, 0)
        self.module = getattr(torch, self.device.type)
        self.module.set_device(self.device.index)
        self.activity, self.kind = PROFILED[self.device.type]

    def sync(self) -> None:
        self.module.synchronize(self.device)

    def name(self) -> str:
        return self.module.get_device_name(self.device.index)


def kernel_ms(dev: Device, fn, iters: int = 5) -> float:
    for _ in range(2):
        fn()
    dev.sync()
    with torch.profiler.profile(activities=[dev.activity]) as prof:
        for _ in range(iters):
            fn()
        dev.sync()
    us = sum(
        e.device_time_total for e in prof.key_averages() if e.device_type == dev.kind
    )
    if us > 0:
        return us / iters / 1e3
    # profiler recorded no device activity: the tuner's device-time measure
    return graph_ms(fn, dev.device)


def quantize(w, fmt):
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    if fmt == "int8":
        s = amax / 127.0
        return torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8), s.float()
    s = amax / 448.0
    return (w / s[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn), s.float()


def arm_times(dev: Device, fn, x, g):
    xa = x.clone().requires_grad_(True)

    def fwd():
        with torch.no_grad():
            fn(x)

    def fwdbwd():
        xa.grad = None
        fn(xa).backward(g)

    return kernel_ms(dev, fwd), kernel_ms(dev, fwdbwd)


def race(dev: Device, m, n, k, dtype, fmt) -> dict:
    """{table: {kernel, eager, compile, config}} from the tuner rows of this shape."""
    prefix = f"m{max(32, 1 << (m - 1).bit_length())}_n{n}_k{k}_{dtype}"
    out = {}
    for kn in TABLES:
        for key, row in tuning.table(dev.device, kn).rows.items():
            if not key.startswith(prefix) or ("float8" in key) != (fmt == "fp8"):
                continue
            t = row["timings_ms"]
            kern = [v for c, v in t.items() if not c.startswith("impl=")]
            out[kn] = {
                "kernel": min(kern) if kern else None,
                "eager": t.get("impl=eager"),
                "compile": t.get("impl=compile"),
                "config": row["config"],
            }
    return out


def _fmt_race(r: dict) -> str:
    def f(v):
        return "  -  " if v is None else f"{v:.3f}"

    return " ".join(
        f"[{kn} k={f(v['kernel'])} e={f(v['eager'])} c={f(v['compile'])}"
        f" -> {v['config'].get('impl', 'kernel')}]"
        for kn, v in r.items()
    )


def run_case(dev: Device, dtype, fmt, name, n, k, m, backends, unavailable):
    d = dev.device
    w = torch.randn(n, k, device=d) * 0.05
    q, scale = quantize(w, fmt)
    x = torch.randn(m, k, device=d, dtype=dtype)
    g = torch.randn(m, n, device=d, dtype=dtype)
    ref = x.double() @ (q.double() * scale.double()[:, None]).T

    def eager(t):
        return quant_linear_reference(t, q, scale, None, fmt)

    arms = {"eager": eager, "compile": torch.compile(eager, dynamic=False)}
    for be in backends:
        op = registry.resolve("quant_linear", d.type, dtype, backend=be, fmt=fmt)
        arms[be] = lambda t, op=op: op(t, q, scale, None, fmt)
    res = {}
    for arm, fn in arms.items():
        try:
            with torch.no_grad():
                got = fn(x)
            err = ((got.double() - ref).abs().max() / ref.abs().max()).item()
            res[arm] = (*arm_times(dev, fn, x, g), err, None)
        except Exception as exc:  # noqa: BLE001 - a failing arm is recorded
            res[arm] = (None, None, None, f"{type(exc).__name__}: {exc}"[:200])
    for be, why in unavailable.items():
        res[f"backend:{be}"] = (None, None, None, f"unavailable: {why}")
    rc = race(dev, m, n, k, dtype, fmt)
    base = [res[a][1] for a in ("eager", "compile") if res[a][1]]
    base_b = min(base) if base else None
    kernel_win = {
        kn: v["kernel"] is not None
        and v["kernel"] < min(x_ for x_ in (v["eager"], v["compile"]) if x_)
        for kn, v in rc.items()
        if v["eager"] or v["compile"]
    }
    sel = [res[be][1] for be in backends if res[be][1]]
    sel_ok = None if not (sel and base_b) else min(sel) <= base_b * 1.03
    verdict = {"kernel": kernel_win, "selected": sel_ok}
    print(
        f"{fmt:5s} {str(dtype)[6:]:8s} {name:8s} m={m:6d} "
        + " ".join(
            f"{a}=" + ("ERR" if v[1] is None else f"{v[0]:.3f}/{v[1]:.3f}")
            for a, v in res.items()
        )
        + f" | selected {'OK' if sel_ok else 'SLOW' if sel_ok is False else '-'}"
        + f" | race {_fmt_race(rc)}",
        flush=True,
    )
    for a, v in res.items():
        if v[3]:
            print(f"   {a}: {v[3]}", flush=True)
    rows = [
        {
            "fmt": fmt,
            "dtype": str(dtype),
            "case": name,
            "n": n,
            "k": k,
            "m": m,
            "arm": a,
            "fwd_ms": v[0],
            "fwdbwd_ms": v[1],
            "rel_err": v[2],
            "note": v[3],
            "race": rc if a in backends else None,
        }
        for a, v in res.items()
    ]
    return rows, (fmt, str(dtype), name, m, verdict, rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bench/quant")
    ap.add_argument("--device", default="cuda", help="cuda[:i] or xpu[:i]")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shapes", default="", help="comma list of SHAPES names")
    ap.add_argument("--tokens", default="", help="comma list of token counts")
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 512
    dev = Device(args.device)
    arch = arch_of(dev.device)
    dtypes = [torch.float16] + ([torch.bfloat16] if arch.has_bf16 else [])
    fmts = ["int8", "fp8"]
    shapes = {"q": SHAPES["q"]} if args.quick else SHAPES
    tokens = (4096,) if args.quick else TOKENS
    if args.shapes:
        shapes = {s: SHAPES[s] for s in args.shapes.split(",")}
    if args.tokens:
        tokens = tuple(int(t) for t in args.tokens.split(","))
    rows, verdicts = [], []
    for fmt in fmts:
        for dtype in dtypes:
            avail = registry.available("quant_linear", dev.device.type, dtype, fmt=fmt)
            avail.pop("eager", None)
            backends = [b for b, why in avail.items() if why is None]
            unavailable = {b: why for b, why in avail.items() if why is not None}
            if not backends:
                print(f"{fmt} {dtype}: no backend serves it: {unavailable}")
                rows.append(
                    {"fmt": fmt, "dtype": str(dtype), "unavailable": unavailable}
                )
                continue
            for name, (n, k) in shapes.items():
                for m in tokens:
                    r, v = run_case(
                        dev, dtype, fmt, name, n, k, m, backends, unavailable
                    )
                    rows += r
                    verdicts.append(v)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    card = dev.name().replace(" ", "_")
    with open(out / f"quant_{card}.json", "w", encoding="utf-8") as fh:
        json.dump({"device": card, "rows": rows, "verdicts": verdicts}, fh, indent=1)
    slow = [v[:4] for v in verdicts if v[4]["selected"] is False]
    print(f"selected path slower than the best PyTorch path in {len(slow)}: {slow}")


if __name__ == "__main__":
    main()
