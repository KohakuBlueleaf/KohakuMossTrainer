"""Bench the optimizer step ops against torch.optim.AdamW (fused and foreach) at MOSS
sizes: a LoRA r32 adapter set (~74M parameters in 504 tensors) and one 50M tensor.

    python scripts/bench/kernel/optim.py --backend triton
"""

import argparse
import json
from pathlib import Path

import torch

from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.hardware.device import measure_rates
from kohakumoss_kernels.optim import new_8bit_state, resolve
from kohakumoss_kernels.tuning.timing import best_ms
from kohakumoss_kernels.tuning.tuner import reset


def lora_shapes():
    """LoRA r32 (down, up) shapes of the q / k / v / o, FFN up and FFN down linears of
    42 blocks."""
    shapes = []
    for _ in range(42):
        for o, i in ((3072, 3072),) * 4 + ((16384, 3072), (3072, 8192)):
            shapes += [(32, i), (o, 32)]
    return shapes


def chosen(op) -> str:
    """`backend` of a registry op, plus `:kernel` / `:torch` for the implementations
    its per-problem choice bound since the last tuner reset."""
    choice = getattr(op, "fn", op)
    impls = getattr(choice, "impls", None)
    if impls is None:
        return op.backend
    names = {id(f): name for name, f in impls.items()}
    picks = sorted({names.get(id(f), "?") for f in choice.bound.values()})
    return f"{op.backend}:{','.join(picks) or 'undecided'}"


def run_case(name, shapes, pdtype, sdtype, mode, backend, device, iters):
    """One row: ms per step of every op and of torch AdamW, and what each op bound."""
    reset()
    ops = resolve(device, pdtype, None if backend == "auto" else backend)
    params = [torch.randn(s, device=device).to(pdtype) for s in shapes]
    grads = [0.01 * torch.randn_like(p) for p in params]
    ms = [torch.zeros_like(p, dtype=sdtype) for p in params]
    rs = [torch.zeros_like(p, dtype=sdtype) for p in params]
    cs = [torch.zeros_like(p) if mode == "kahan" else None for p in params]
    q8 = [new_8bit_state(p) for p in params]
    step = [0]

    def ours():
        step[0] += 1
        for p, g, m, r, c in zip(params, grads, ms, rs, cs):
            ops.adamw(
                p, g, m, r, c, 1e-4, 0.9, 0.95, 1e-8, 0.01, step[0], mode, step[0]
            )

    def ours8():
        step[0] += 1
        for p, g, s, c in zip(params, grads, q8, cs):
            ops.adamw8bit(
                p, g, *s, c, 1e-4, 0.9, 0.95, 1e-8, 0.01, step[0], mode, step[0]
            )

    def ours_multi():
        step[0] += 1
        ops.adamw_multi(
            params, grads, ms, rs, cs if mode == "kahan" else None,
            1e-4, 0.9, 0.95, 1e-8, 0.01, step[0], mode, step[0],
        )  # fmt: skip

    row = {"case": name, "param": str(pdtype), "state": str(sdtype), "mode": mode}
    row["ours_ms"] = best_ms(ours, device, iters)
    row["ours_multi_ms"] = best_ms(ours_multi, device, iters)
    row["ours8bit_ms"] = best_ms(ours8, device, iters)
    row["ours_impl"] = chosen(ops.adamw)
    row["ours_multi_impl"] = chosen(ops.adamw_multi)
    row["ours8bit_impl"] = chosen(ops.adamw8bit)
    if pdtype == torch.float32 or mode == "nearest":
        tparams = [torch.nn.Parameter(p.detach().float().clone()) for p in params]
        for p, g in zip(tparams, grads):
            p.grad = g.float()
        for flag in ("fused", "foreach"):
            try:
                opt = torch.optim.AdamW(
                    tparams, lr=1e-4, betas=(0.9, 0.95), **{flag: True}
                )
                row[f"torch_{flag}_fp32_ms"] = best_ms(opt.step, device, iters)
            except RuntimeError as err:
                row[f"torch_{flag}_fp32_error"] = str(err)[:200]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="auto")
    p.add_argument("--out", default="runs/kernel")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", default="cuda", help="cuda | cuda:N | xpu:N")
    args = p.parse_args()
    device = torch.device(args.device)
    has_bf16 = arch_of(device).has_bf16
    half = torch.bfloat16 if has_bf16 else torch.float16
    cases = [
        ("lora-fp32", lora_shapes(), torch.float32, torch.float32, "nearest"),
        ("big-fp32", [(50_000_000,)], torch.float32, torch.float32, "nearest"),
        ("big-16bit-kahan", [(50_000_000,)], half, half, "kahan"),
    ]
    if has_bf16:
        cases.append(
            (
                "big-bf16-sr",
                [(50_000_000,)],
                torch.bfloat16,
                torch.bfloat16,
                "stochastic",
            )
        )
    rows = [run_case(*c, args.backend, device, args.iters) for c in cases]
    card = measure_rates(device).key
    path = Path(args.out) / f"optim-{card}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"card": card, "rows": rows}, indent=1), encoding="utf-8"
    )
    for r in rows:
        print(
            " | ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r.items()
            )
        )


if __name__ == "__main__":
    main()
