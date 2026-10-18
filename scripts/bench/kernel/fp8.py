"""fp8 / MXFP8 compute linear vs bf16 / fp16 F.linear: every GEMM path of every recipe.

Per (recipe, dtype, layer shape, tokens): the 16-bit `F.linear` (and `dy @ W`) baseline,
each path's fastest tile (activation quantizer included), the path the tuner selects,
for the forward and the fp8 input gradient, plus the selected path's error against the
fp64 product with the quantized weight. Times are the tuner's own device measure (best
of 10 after warmup, host launch gaps included). One JSON per card.

    python scripts/bench/kernel/fp8.py --out runs/bench/fp8
    python scripts/bench/kernel/fp8.py --quick --recipes mx
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from kohakumoss.quant.compute import quantize_pair
from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.triton.fp8 import linear as fp8_linear
from kohakumoss_kernels.fp8 import RECIPES, dequantize
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.tuning.timing import best_ms

SHAPES = {
    "dit_q": (3072, 3072),
    "dit_up_gate": (16384, 3072),
    "dit_down": (3072, 8192),
    "te_q": (8192, 4096),
    "te_up": (12288, 4096),
    "te_down": (4096, 12288),
}
TOKENS = (16, 256, 1024, 4096, 16384)


def _name(cfg: dict) -> str:
    return cfg["gemm"] + (f"-{cfg['impl']}" if "impl" in cfg else "")


def path_times(runner, recipe: str, m: int, k: int, sm: int, args: tuple, dev) -> dict:
    """{path: best ms over its tiles} of one direction (`k` the contraction length)."""
    out = {}
    for cfg in fp8_linear.candidates(recipe, sm, m, True, k):
        if cfg["gemm"] == "w16" and cfg["impl"] == "eager":
            continue
        try:
            fn = runner(cfg, recipe)
            ms = best_ms(lambda fn=fn: fn(*args), dev, 10)
        except Exception as exc:  # noqa: BLE001 - a failing path is recorded
            out.setdefault("failed", {})[str(cfg)] = f"{type(exc).__name__}"[:80]
            continue
        name = _name(cfg)
        out[name] = min(ms, out.get(name, ms))
    return out


def run_case(dev, sm, recipe, dtype, name, n, k, m) -> dict:
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(m, k, device=dev, generator=g).to(dtype)
    w = (torch.randn(n, k, device=dev, generator=g) * 0.05).to(dtype)
    dy = torch.randn(m, n, device=dev, generator=g).to(dtype)
    q, s, aux = quantize_pair(w, recipe)
    qt, st, auxt = quantize_pair(w.t(), recipe)
    fwd_args = (x, q, s, aux, None)
    fwd = path_times(fp8_linear._fwd_runner, recipe, m, k, sm, fwd_args, dev)
    bwd_args = (dy, q, s, qt, st, auxt)
    bwd = path_times(fp8_linear._dgrad_runner, recipe, m, n, sm, bwd_args, dev)
    base_f = best_ms(lambda: F.linear(x, w), dev, 10)
    base_b = best_ms(lambda: dy @ w, dev, 10)
    y = fp8_linear.fwd(x, q, s, aux, None, recipe)
    key = fp8_linear._table_key("fwd", m, n, k, x, recipe, None, True)
    pick = tuning.table("fp8_fwd", dev).get(key) or {}
    ref = x.double() @ dequantize(q, s, recipe, torch.float64).T
    err = ((y.double() - ref).abs().mean() / ref.abs().mean()).item()
    best_f = min(v for p, v in fwd.items() if p != "failed")
    best_b = min(v for p, v in bwd.items() if p != "failed")
    row = {
        "recipe": recipe, "dtype": str(dtype)[6:], "case": name, "m": m, "n": n, "k": k,
        "bf16_fwd_ms": base_f, "bf16_dgrad_ms": base_b, "fwd": fwd, "dgrad": bwd,
        "speedup_fwd": base_f / best_f, "speedup_dgrad": base_b / best_b,
        "selected_fwd": pick, "rel_err_mean": err,
    }  # fmt: skip
    print(
        f"{recipe:6s} {row['dtype']:8s} {name:12s} m={m:6d} 16bit {base_f:.3f}/"
        f"{base_b:.3f} best {best_f:.3f}/{best_b:.3f} x{row['speedup_fwd']:.2f}/"
        f"x{row['speedup_dgrad']:.2f} " + " ".join(f"{p}={v:.3f}" for p, v in fwd.items() if p != "failed"),  # fmt: skip
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bench/fp8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--recipes", default=",".join(RECIPES))
    ap.add_argument("--shapes", default="", help="comma list of SHAPES names")
    ap.add_argument("--tokens", default="", help="comma list of token counts")
    args = ap.parse_args()
    dev = torch.device(args.device)
    dev = torch.device(dev.type, dev.index or 0)
    torch.cuda.set_device(dev)
    arch = arch_of(dev)
    if not arch.has_fp8:
        raise SystemExit(f"{arch.name} has no fp8 GEMMs (sm89+)")
    dtypes = [torch.bfloat16, torch.float16]
    shapes = SHAPES if not args.quick else {"dit_q": SHAPES["dit_q"]}
    tokens = TOKENS if not args.quick else (256, 4096)
    if args.shapes:
        shapes = {s: SHAPES[s] for s in args.shapes.split(",")}
    if args.tokens:
        tokens = tuple(int(t) for t in args.tokens.split(","))
    rows = []
    for recipe in args.recipes.split(","):
        for dtype in dtypes:
            for name, (n, k) in shapes.items():
                for m in tokens:
                    rows.append(run_case(dev, arch.sm, recipe, dtype, name, n, k, m))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    card = torch.cuda.get_device_name(dev).replace(" ", "_")
    path = out / f"fp8_{card}.json"
    path.write_text(json.dumps({"device": card, "rows": rows}, indent=1), "utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
