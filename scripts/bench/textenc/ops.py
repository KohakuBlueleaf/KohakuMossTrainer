"""Text-encoder op timing table: per shape (segments x length) the best config of every
design of `gated_delta_rule_varlen` / `attention_varlen_causal` on this device, the
eager reference, the transformers PyTorch reference and, when installed (the
`baselines` extra), flash-linear-attention; error stats of each fastest config against
fp64.

    python scripts/bench/textenc/ops.py --shapes 1x300,16x300,64x150,256x40,1x2048 \
        --dtype bf16 --out runs/bench/textenc

Shapes use the Qwen3.5-9B head layout (delta: 16 key / 32 value heads of 128;
attention: 16 query / 4 kv heads of 256). Writes <out>/<gpu>_ops_<dtype>.json.
"""

import argparse
import importlib
import json
import os
import time

import torch
import torch.nn.functional as F

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import textenc as E

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _optional(name):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None


def ms(fn, iters: int = 10) -> float:
    fn()
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def rel_rms(out, ref) -> float:
    d = out.double() - ref
    return (d.square().mean().sqrt() / ref.square().mean().sqrt()).item()


def delta_case(nseg, length, dtype, dev):
    n = nseg * length
    q = F.silu(torch.randn(n, 16, 128, device=dev)).to(dtype)
    k = F.silu(torch.randn(n, 16, 128, device=dev)).to(dtype)
    v = F.silu(torch.randn(n, 32, 128, device=dev)).to(dtype)
    a = (torch.randn(n, 32, device=dev) * 2).to(dtype)
    b = (torch.randn(n, 32, device=dev) * 2).to(dtype)
    a_log = torch.log(torch.rand(32, device=dev) * 16 + 0.01)
    dt_bias = torch.randn(32, device=dev)
    cu = torch.arange(nseg + 1, device=dev, dtype=torch.int32) * length
    return q, k, v, a, b, a_log, dt_bias, cu, length


def selected(row: dict, op: str, args, dev, **static) -> None:
    """Time each usable non-eager backend's registry entry for `op` on `dev` (the config
    its own timed shortlist selects, as the encoder gets it)."""
    dtype = args[0].dtype
    for name, why in registry.available(op, dev, dtype, **static).items():
        if name == "eager" or why is not None:
            continue
        fn = registry.resolve(op, dev, dtype, backend=name, **static)
        row[f"selected_{name}_ms"] = ms(lambda: fn(*args))


def bench_delta(nseg, length, dtype, dev, backends) -> dict:
    args = delta_case(nseg, length, dtype, dev)
    ref = E.gated_delta_rule_varlen(*(t.double() for t in args[:7]), *args[7:])
    row = {"op": "gated_delta_rule_varlen", "shape": f"{nseg}x{length}"}
    row["eager_ms"] = ms(lambda: E.gated_delta_rule_varlen(*args), 3)
    selected(row, "gated_delta_rule_varlen", args, dev, head_dim_k=128, head_dim_v=128)
    for name, (run, candidates) in backends.items():
        best = {}
        for cfg in candidates(dev, dtype):
            try:
                t = ms(lambda: run(cfg, *args))
            except Exception:  # noqa: BLE001
                continue
            design = cfg.get("impl", name)
            if design not in best or t < best[design][0]:
                best[design] = (t, cfg)
        for design, (t, cfg) in best.items():
            row[f"{name}_{design}_ms"] = t
            row[f"{name}_{design}_cfg"] = cfg
            row[f"{name}_{design}_rel"] = rel_rms(run(cfg, *args), ref)
    hf = _optional("transformers.models.qwen3_5.modeling_qwen3_5")
    q, k, v, a, b, a_log, dt_bias, cu, _ = args
    g = -a_log.exp() * F.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b.float()).to(dtype)
    qr, kr = q.repeat_interleave(2, 1), k.repeat_interleave(2, 1)
    if hf is not None:
        torch_chunk = getattr(
            hf.torch_chunk_gated_delta_rule,
            "__wrapped__",
            hf.torch_chunk_gated_delta_rule,
        )
        view = [t.view(nseg, length, *t.shape[1:]) for t in (qr, kr, v, g, beta)]
        row["hf_torch_ms"] = ms(
            lambda: torch_chunk(*view[:3], g=view[3], beta=view[4], use_qk_l2norm_in_kernel=True),  # fmt: skip
            3,
        )
    fla = _optional("fla.ops.gated_delta_rule")
    if fla is not None:

        def run_fla():
            return fla.chunk_gated_delta_rule(
                qr[None], kr[None], v[None], g[None], beta[None],
                cu_seqlens=cu.long(), use_qk_l2norm_in_kernel=True,
            )[0][0]  # fmt: skip

        row["fla_ms"] = ms(run_fla)
        row["fla_rel"] = rel_rms(run_fla(), ref)
    return row


def bench_attention(nseg, length, dtype, dev, backends) -> dict:
    n = nseg * length
    q = torch.randn(n, 16, 256, device=dev).to(dtype)
    k = torch.randn(n, 4, 256, device=dev).to(dtype)
    v = torch.randn(n, 4, 256, device=dev).to(dtype)
    cu = torch.arange(nseg + 1, device=dev, dtype=torch.int32) * length
    args = (q, k, v, cu, length, 256**-0.5)
    ref = E.attention_varlen_causal(q.double(), k.double(), v.double(), *args[3:])
    row = {"op": "attention_varlen_causal", "shape": f"{nseg}x{length}"}
    row["eager_sdpa_ms"] = ms(lambda: E.attention_varlen_causal(*args), 3)
    selected(row, "attention_varlen_causal", args, dev, head_dim=256)
    for name, (run, candidates) in backends.items():
        best = {}
        for cfg in candidates(dev, dtype, 256):
            try:
                t = ms(lambda: run(cfg, *args))
            except Exception:  # noqa: BLE001
                continue
            design = cfg.get("impl", name)
            if design not in best or t < best[design][0]:
                best[design] = (t, cfg)
        for design, (t, cfg) in best.items():
            row[f"{name}_{design}_ms"] = t
            row[f"{name}_{design}_cfg"] = cfg
            row[f"{name}_{design}_rel"] = rel_rms(run(cfg, *args), ref)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1x300,16x300,64x150,256x40,1x2048,4x1024")
    ap.add_argument("--dtype", default="bf16", choices=tuple(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/bench/textenc")
    ap.add_argument(
        "--all-backends",
        action="store_true",
        help="also sweep backends the registry does not serve on this device",
    )
    args = ap.parse_args()
    dev = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    registry.load_backends()
    delta, attn = {}, {}
    tri = _optional("kohakumoss_kernels.backends.triton.textenc.delta")
    if tri is not None and dev.type in ("cuda", "xpu"):
        delta["triton"] = (tri.run, tri.candidates)
    tl = _optional("kohakumoss_kernels.backends.tilelang.textenc.ops")
    if tl is not None and dev.type == "cuda":
        delta["tilelang"] = (tl.delta_run, lambda d, t: tl.delta_candidates())
        if dtype != torch.float32:
            attn["tilelang"] = (tl.attn_run, lambda d, t, h: tl.attn_candidates(h))
    tra = _optional("kohakumoss_kernels.backends.triton.textenc.attention")
    if tra is not None and dtype != torch.float32:
        attn["triton"] = (tra.run, tra.candidates)
    if not args.all_backends:
        ok = {
            op: {
                b
                for b, why in registry.available(op, dev, dtype, **st).items()
                if why is None
            }
            for op, st in (
                ("gated_delta_rule_varlen", {"head_dim_k": 128, "head_dim_v": 128}),
                ("attention_varlen_causal", {"head_dim": 256}),
            )
        }
        delta = {b: v for b, v in delta.items() if b in ok["gated_delta_rule_varlen"]}
        attn = {b: v for b, v in attn.items() if b in ok["attention_varlen_causal"]}
    rows = []
    for spec in args.shapes.split(","):
        nseg, length = (int(x) for x in spec.split("x"))
        for fn, backends in ((bench_delta, delta), (bench_attention, attn)):
            row = fn(nseg, length, dtype, dev, backends)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("_cfg")}), flush=True)  # fmt: skip
    name = getattr(torch, dev.type).get_device_name(dev)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{name.replace(' ', '_')}_ops_{args.dtype}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"device": name, "args": vars(args), "rows": rows}, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
