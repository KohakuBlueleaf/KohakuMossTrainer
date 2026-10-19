"""ConvRot rotation cost vs the int8 linear it feeds, and fused vs separate.

Per DiT shape (N, K) and token count M, on one card, in fp16 and bf16 (where it has
bf16): the rotation alone on x [M, K] (`group_hadamard`, every backend that serves it
plus torch.compile of the eager reference), the plain int8 `quant_linear` (forward and
forward + input gradient), and the registered `quant_linear_rotated` (the path the tuner
selected). The tuner's own device times of each rotated candidate (`separate`, every
fused tile) come from its tables "quant_fwd_rot" / "quant_dgrad_rot". Every row also
carries the error against fp64. One JSON per card.

    python scripts/bench/kernel/rotate.py --out runs/kernel --device cuda
"""

import argparse
import json
from pathlib import Path

import torch

from kohakumoss_kernels import registry, tuning
from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.hadamard import group_hadamard_eager, hadamard_matrix
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.quant.rotated import quant_linear_rotated_reference
from kohakumoss_kernels.tuning.buckets import pow2
from kohakumoss_kernels.tuning.timing import graph_ms

SHAPES = {
    "qkv": (9216, 3072),
    "o": (3072, 3072),
    "up_gate": (16384, 3072),
    "down": (3072, 8192),
}
TOKENS = (4096, 8192, 16384)
GROUP = 256
ROT_TABLES = ("quant_fwd_rot", "quant_dgrad_rot")
PROFILED = {
    "cuda": (torch.profiler.ProfilerActivity.CUDA, torch.autograd.DeviceType.CUDA),
    "xpu": (torch.profiler.ProfilerActivity.XPU, torch.autograd.DeviceType.XPU),
}


class Device:
    """The bench device, its torch device module and profiler activity, resolved once."""

    def __init__(self, name: str):
        self.device = torch.device(name)
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
    """Device time per call (profiler sum of kernels, host dispatch excluded)."""
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
    return us / iters / 1e3 if us > 0 else graph_ms(fn, dev.device)


def quantize(w):
    s = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    return torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8), s.float()


def rel_err(got, ref) -> float:
    return ((got.double() - ref).abs().max() / ref.abs().max()).item()


def rot64(x):
    h = hadamard_matrix(GROUP, torch.float32, x.device).double()
    return (x.double().reshape(-1, GROUP) @ h).reshape(x.shape)


def rotation_rows(dev: Device, x) -> dict:
    """{path: {ms, rel_err}} of the rotation alone, per serving backend."""
    ref = rot64(x)
    avail = registry.available("group_hadamard", dev.device.type, x.dtype)
    paths = {"eager": lambda: group_hadamard_eager(x, GROUP)}
    paths["compile"] = lambda: compiled_op("group_hadamard")(x, GROUP)
    for be in ("triton", "tilelang"):
        if avail.get(be) is None:
            op = registry.resolve("group_hadamard", dev.device, x.dtype, backend=be)
            fn = getattr(op, "__wrapped__", op)
            paths[be] = lambda fn=fn: fn(x, GROUP)
    out = {}
    with torch.no_grad():
        for name, fn in paths.items():
            try:
                err = rel_err(fn(), ref)
                out[name] = {"ms": kernel_ms(dev, fn), "rel_err": err}
            except Exception as exc:  # noqa: BLE001 - a failing path is recorded
                out[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def linear_rows(dev: Device, x, q, scale, g) -> dict:
    """{plain, rotated}: forward and forward + input-gradient times and errors."""
    d, dtype = dev.device, x.dtype
    plain = registry.resolve("quant_linear", d, dtype, fmt="int8")
    rotated = registry.resolve(
        "quant_linear_rotated", d, dtype, fmt="int8", group=GROUP
    )
    w64 = q.double() * scale.double()[:, None]
    refs = {"plain": x.double() @ w64.T, "rotated": rot64(x) @ w64.T}
    fns = {
        "plain": lambda t: plain(t, q, scale, None, "int8"),
        "rotated": lambda t: rotated(t, q, scale, None, "int8", GROUP),
        "rotated_eager": lambda t: quant_linear_rotated_reference(
            t, q, scale, None, "int8", GROUP
        ),
    }
    refs["rotated_eager"] = refs["rotated"]
    out = {}
    for name, fn in fns.items():
        xa = x.clone().requires_grad_(True)

        def fwd(fn=fn):
            with torch.no_grad():
                fn(x)

        def fwdbwd(fn=fn, xa=xa):
            xa.grad = None
            fn(xa).backward(g)

        try:
            with torch.no_grad():
                err = rel_err(fn(x), refs[name])
            out[name] = {
                "fwd_ms": kernel_ms(dev, fwd),
                "fwdbwd_ms": kernel_ms(dev, fwdbwd),
                "rel_err": err,
                "backend": getattr(
                    {"plain": plain, "rotated": rotated}.get(name), "backend", "eager"
                ),
            }
        except Exception as exc:  # noqa: BLE001 - a failing path is recorded
            out[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def tuner_rows(dev: Device, m, n, k, dtype) -> dict:
    """{table: {config: ms, chosen}} of the rotated candidates of this shape."""
    prefix = f"m{pow2(m, 32)}_n{n}_k{k}_g{GROUP}_{dtype}"
    out = {}
    for kn in ROT_TABLES:
        for key, row in tuning.table(kn, dev.device).rows.items():
            if key.startswith(prefix):
                out[kn] = {"timings_ms": row["timings_ms"], "chosen": row["config"]}
    return out


def run_case(dev: Device, dtype, name, n, k, m) -> dict:
    d = dev.device
    q, scale = quantize(torch.randn(n, k, device=d) * 0.05)
    x = torch.randn(m, k, device=d, dtype=dtype)
    g = torch.randn(m, n, device=d, dtype=dtype)
    rot = rotation_rows(dev, x)
    lin = linear_rows(dev, x, q, scale, g)
    tun = tuner_rows(dev, m, n, k, dtype)
    best_rot = min((v["ms"] for v in rot.values() if "ms" in v), default=None)
    plain = lin.get("plain", {}).get("fwd_ms")
    rotated = lin.get("rotated", {}).get("fwd_ms")
    row = {
        "dtype": str(dtype),
        "case": name,
        "n": n,
        "k": k,
        "m": m,
        "rotation": rot,
        "linear": lin,
        "tuner": tun,
        "rotate_over_quant_fwd": None if not (best_rot and plain) else best_rot / plain,
        "rotated_over_plain_fwd": None if not (rotated and plain) else rotated / plain,
    }
    pb = lin.get("plain", {}).get("fwdbwd_ms")
    rb = lin.get("rotated", {}).get("fwdbwd_ms")
    row["rotated_over_plain_fwdbwd"] = None if not (pb and rb) else rb / pb
    print(
        f"{str(dtype)[6:]:8s} {name:8s} m={m:6d} "
        + " ".join(f"rot.{a}={v.get('ms', float('nan')):.3f}" for a, v in rot.items())
        + f" | plain {plain} rotated {rotated} | x{row['rotated_over_plain_fwd']}"
        + f" fwdbwd x{row['rotated_over_plain_fwdbwd']}"
        + " | "
        + " ".join(
            f"[{t}: {v['chosen'].get('impl')} {v['timings_ms']}]"
            for t, v in tun.items()
        ),
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/kernel")
    ap.add_argument("--device", default="cuda", help="cuda[:i] or xpu[:i]")
    ap.add_argument("--shapes", default="", help="comma list of SHAPES names")
    ap.add_argument("--tokens", default="", help="comma list of token counts")
    args = ap.parse_args()
    torch._dynamo.config.recompile_limit = 512
    dev = Device(args.device)
    arch = arch_of(dev.device)
    dtypes = [torch.float16] + ([torch.bfloat16] if arch.has_bf16 else [])
    shapes = SHAPES
    if args.shapes:
        shapes = {s: SHAPES[s] for s in args.shapes.split(",")}
    tokens = tuple(int(t) for t in args.tokens.split(",")) if args.tokens else TOKENS
    rows = []
    for dtype in dtypes:
        for name, (n, k) in shapes.items():
            for m in tokens:
                rows.append(run_case(dev, dtype, name, n, k, m))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    card = dev.name().replace(" ", "_")
    with open(out / f"rotate_{card}.json", "w", encoding="utf-8") as fh:
        json.dump({"device": card, "arch": arch.name, "rows": rows}, fh, indent=1)
    print(f"wrote {out / f'rotate_{card}.json'}")


if __name__ == "__main__":
    main()
