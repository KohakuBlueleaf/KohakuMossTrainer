"""Write an fp16-safe MOSS-Image checkpoint: same keys and shapes, stock eps.

The uniform residual fold c (and optional pre-gate scale s) is chosen from an activation
probe report (or given) and written into the weights; the factors are stored in the file
metadata. `EPS_CHECK` measures, on the real weights in fp32, how far the stock eps is
from the exact eps * c^2.

    kogine run scripts/export/fp16_fix.py --set WEIGHTS=<root>/base \
        --set PROBE=activations.json --set OUT=moss_image_base_fp16fix.safetensors
    kogine run scripts/export/fp16_fix.py --set WEIGHTS=<root>/base --set C=0.0625 \
        --set EPS_CHECK=true --set EMBEDS=embeds.pt --set DEVICE=cuda --set OUT=fix.safetensors
"""

import json
from pathlib import Path

import torch

from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.io import save_single
from kohakumoss.model.numerics import choose_factors, fold, unfold
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_common.knobs import check_script

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
log = get_logger("scripts.fp16_fix")

WEIGHTS = ""  # HF transformer dir / root or a single file
PROBE = ""  # activations.json from scripts/probe/activations.py; "" = factors below
TARGET = 2.0**12  # max |x| of the residual stream after the fix
C: float | None = None  # fold factor c; None = from PROBE (or 1.0)
S_ATTN: float | None = None  # attention pre-gate scale; None = from PROBE (or 1.0)
S_FFN: float | None = None  # FFN pre-gate scale; None = from PROBE (or 1.0)
SAVE_DTYPE = "bf16"  # "bf16" | "fp16" | "fp32"
FUSED_QKV = False  # write joint q/k/v as one proj.qkv
EPS_CHECK = False  # also measure stock vs exact eps on the real weights (fp32)
EMBEDS = ""  # EPS_CHECK: embeddings file from scripts/probe/embed.py
DEVICE = "cuda"  # EPS_CHECK device
OUT = ""  # output .safetensors (report next to it as .json)


def eps_check(weights: str, embeds: str, device: str, factors: dict) -> list[dict]:
    """Output error of the stock-eps fold vs the exact-eps fold (fp32, real weights)."""
    dev = torch.device(device)
    data = torch.load(embeds)
    model, _ = load_backbone(weights, device=dev, dtype=torch.float32)
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
            log.info(str(rows[-1]))
    del model
    return rows


def _rel(a, b) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not OUT:
        raise SystemExit("set WEIGHTS and OUT")
    factors = {"c": 1.0, "s_attn": 1.0, "s_ffn": 1.0}
    probe = {}
    if PROBE:
        probe = json.loads(Path(PROBE).read_text(encoding="utf-8"))
        factors = choose_factors(probe, TARGET)
    for key, value in (("c", C), ("s_attn", S_ATTN), ("s_ffn", S_FFN)):
        if value is not None:
            factors[key] = value
    report = {"factors": factors, "target": TARGET}
    if probe:
        report["probe_summary_max"] = probe["summary_max"]
    if EPS_CHECK:
        report["eps_check"] = eps_check(WEIGHTS, EMBEDS, DEVICE, factors)
    model, load = load_backbone(WEIGHTS, device="cpu", dtype=torch.bfloat16)
    fold(model, factors["c"], "gate", (factors["s_attn"], factors["s_ffn"]), "stock")
    meta = {"source": Path(WEIGHTS).name, "target": str(TARGET)}
    save_single(
        model,
        OUT,
        dtype=DTYPES[SAVE_DTYPE],
        fused_qkv=FUSED_QKV,
        metadata=meta,
        keep_fold=True,
    )
    report.update(
        out=str(OUT),
        keys=load.n_source,
        bytes=Path(OUT).stat().st_size,
        save_dtype=SAVE_DTYPE,
        fused_qkv=FUSED_QKV,
    )
    Path(OUT).with_suffix(".json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    log.info(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
