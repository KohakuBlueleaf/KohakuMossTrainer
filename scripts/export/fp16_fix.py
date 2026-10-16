"""Write an fp16-safe MOSS-Image checkpoint: same keys and shapes, stock eps.

The uniform residual fold c (and optional pre-gate scale s) is chosen from an activation
probe report (or given) and written into the weights; the factors are stored in the file
metadata. `--eps-check` measures, on the real weights in fp32, how far the stock eps is
from the exact eps * c^2.

    python scripts/export/fp16_fix.py --weights <root>/base --probe activations.json \
        --out moss_image_base_fp16fix.safetensors
    python scripts/export/fp16_fix.py --weights <root>/base --c 0.0625 --eps-check \
        --embeds embeds.pt --device cuda --out fix.safetensors
"""

import argparse
import json
from pathlib import Path

import torch

from kohakumoss.model import load_backbone
from kohakumoss.model.io import save_single
from kohakumoss.model.numerics import choose_factors, fold, unfold
from kohakumoss.types.seqinfo import build_seqinfo

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def eps_check(args, factors) -> list[dict]:
    """Output error of the stock-eps fold vs the exact-eps fold (fp32, real weights)."""
    dev = torch.device(args.device)
    data = torch.load(args.embeds)
    model, _ = load_backbone(args.weights, device=dev, dtype=torch.float32)
    model.requires_grad_(False)
    rows = []
    cases = [(64, 64, 0.9), (64, 64, 0.3), (64, 64, 0.02), (32, 32, 0.5)]
    with torch.inference_mode():
        for h, w, sigma in cases:
            g = torch.Generator().manual_seed(h + int(100 * sigma))
            img = torch.randn((h * w, 128), generator=g).to(dev)
            text = data["embeds"][0].to(dev)
            seq = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([text.shape[0]]))
            seq = seq.to(dev)
            s = torch.tensor([sigma], device=dev)
            ref = model(img, text, s, seq)
            outs = {}
            pregate = (factors["s_attn"], factors["s_ffn"])
            for eps in ("exact", "stock"):
                fold(model, factors["c"], "gate", pregate, eps)
                outs[eps] = model(img, text, s, seq)
                unfold(model)
            rows.append(
                {
                    "hw": [h, w],
                    "sigma": sigma,
                    "exact_vs_unfolded": _rel(outs["exact"], ref),
                    "stock_vs_unfolded": _rel(outs["stock"], ref),
                }
            )
            print(rows[-1], flush=True)
    del model
    return rows


def _rel(a, b) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="HF transformer dir/root or file")
    ap.add_argument("--probe", default="", help="activations.json from the fp16 probe")
    ap.add_argument("--target", type=float, default=2.0**12, help="max |x| after fix")
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--s-attn", type=float, default=None)
    ap.add_argument("--s-ffn", type=float, default=None)
    ap.add_argument("--save-dtype", default="bf16", choices=tuple(DTYPES))
    ap.add_argument("--fused-qkv", action="store_true")
    ap.add_argument("--eps-check", action="store_true")
    ap.add_argument("--embeds", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    factors = {"c": 1.0, "s_attn": 1.0, "s_ffn": 1.0}
    probe = {}
    if args.probe:
        probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
        factors = choose_factors(probe, args.target)
    for key, value in (("c", args.c), ("s_attn", args.s_attn), ("s_ffn", args.s_ffn)):
        if value is not None:
            factors[key] = value
    report = {"factors": factors, "target": args.target}
    if probe:
        report["probe_summary_max"] = probe["summary_max"]
    if args.eps_check:
        report["eps_check"] = eps_check(args, factors)
    model, load = load_backbone(args.weights, device="cpu", dtype=torch.bfloat16)
    fold(model, factors["c"], "gate", (factors["s_attn"], factors["s_ffn"]), "stock")
    meta = {"source": Path(args.weights).name, "target": str(args.target)}
    save_single(
        model,
        args.out,
        dtype=DTYPES[args.save_dtype],
        fused_qkv=args.fused_qkv,
        metadata=meta,
        keep_fold=True,
    )
    report.update(
        out=str(args.out),
        keys=load.n_source,
        bytes=Path(args.out).stat().st_size,
        save_dtype=args.save_dtype,
        fused_qkv=args.fused_qkv,
    )
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
