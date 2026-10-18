"""fp16 backward probe: per-module forward |out| max and backward |grad of out| max on the
real backbone, and the first module whose output gradient is non-finite.

The latent and text inputs require grad (the base stays frozen), so every module output
on the path gets a gradient; the loss is the training MSE times each of LOSS_SCALES.
Writes a JSON report.

    kogine run scripts/probe/backward.py --set WEIGHTS=<fp16-fix file> \
        --set OUT=runs/probe/backward.json
"""

import json
import math
from pathlib import Path

import torch

from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.numerics.fp16fix import apply_fp16_fix
from kohakumoss.types.plans import PrecisionConfig
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.backward")

WEIGHTS = (
    ""  # transformer dir / root or single file (the fp16-fix file for FP16_FIX none)
)
OUT = ""  # output report .json
DEVICE = "cuda"
PRECISION = "fp16"  # compute dtype: "fp16" | "bf16"
FP16_FIX = "none"  # "none" (fix file or bf16) | "online" | "fp32_residual"
FP16_FIX_KWARGS = {}  # e.g. {"c": 0.0625}
KERNELS = {}  # op -> backend overrides, e.g. {"attention_varlen": "eager"}
SIZES = [(1024, 1024)]  # (H, W) per probed image; one report row set per size
TEXT_LEN = 300
SIGMAS = (0.9, 0.5, 0.1)
LOSS_SCALES = (65536.0, 1024.0, 1.0)
SEED = 0

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _first_tensor(out):
    match out:
        case torch.Tensor():
            return out
        case tuple() | list():
            return next((t for t in out if isinstance(t, torch.Tensor)), None)
    return None


class Probe:
    """Forward hooks record each module's output max and hook its gradient."""

    def __init__(self, model):
        self.order: list[str] = []
        self.fwd: dict[str, float] = {}
        self.bwd: dict[str, float] = {}
        self.bad_fwd: list[str] = []
        self.bad_bwd: list[str] = []
        self.handles = [
            m.register_forward_hook(self._hook(name))
            for name, m in model.named_modules()
            if name
        ]

    def reset(self) -> None:
        self.order, self.fwd, self.bwd = [], {}, {}
        self.bad_fwd, self.bad_bwd = [], []

    def _hook(self, name: str):
        def hook(module, args, out):
            t = _first_tensor(out)
            if t is None or not t.is_floating_point():
                return
            v = float(t.detach().abs().float().max())
            self.fwd[name] = max(self.fwd.get(name, 0.0), v)
            if not math.isfinite(v):
                self.bad_fwd.append(name)
            if t.requires_grad:
                t.register_hook(self._grad(name))

        return hook

    def _grad(self, name: str):
        def hook(g):
            v = float(g.detach().abs().float().max())
            self.order.append(name)
            self.bwd[name] = max(self.bwd.get(name, 0.0), v)
            if not math.isfinite(v):
                self.bad_bwd.append(name)

        return hook


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not OUT:
        raise SystemExit("set WEIGHTS and OUT")
    dev = torch.device(DEVICE)
    dtype = DTYPES[PRECISION]
    keep = PrecisionConfig().keep_fp32
    model, _ = load_backbone(
        WEIGHTS, device=dev, dtype=dtype, keep_fp32=keep, kernels=dict(KERNELS)
    )
    apply_fp16_fix(model, FP16_FIX, **dict(FP16_FIX_KWARGS))
    model.requires_grad_(False)
    probe = Probe(model)
    rows = []
    for size in SIZES:
        rows += probe_size(model, probe, dev, dtype, size)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    log.info(f"wrote {OUT}")


def probe_size(model, probe: Probe, dev, dtype, size) -> list[dict]:
    """The report rows of one (H, W) image: every sigma x loss scale."""
    h, w = size[0] // 16, size[1] // 16
    g = torch.Generator(dev).manual_seed(SEED)
    cfg = model.cfg
    z0 = torch.randn((h * w, cfg.latent_channels), generator=g, device=dev)
    noise = torch.randn((h * w, cfg.latent_channels), generator=g, device=dev)
    text0 = torch.randn((TEXT_LEN, cfg.text_dim), generator=g, device=dev)
    seq = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([TEXT_LEN])).to(dev)
    rows = []
    for sigma in SIGMAS:
        for scale in LOSS_SCALES:
            probe.reset()
            x = ((1 - sigma) * z0 + sigma * noise).requires_grad_(True)
            text = text0.to(dtype).requires_grad_(True)
            s = torch.tensor([sigma], device=dev)
            with torch.autocast(dev.type, dtype=dtype):
                pred = model(x, text, s, seq)
            loss = (pred.float() - (noise - z0)).square().mean()
            (loss * scale).backward()
            row = {
                "size": list(size),
                "sigma": sigma,
                "loss_scale": scale,
                "loss": float(loss),
                "nonfinite_forward": probe.bad_fwd[:8],
                "first_nonfinite_grad": probe.bad_bwd[:1],
                "nonfinite_grads": len(probe.bad_bwd),
                "x_grad_finite": bool(torch.isfinite(x.grad).all()),
                "text_grad_finite": bool(torch.isfinite(text.grad).all()),
                "fwd_top": sorted(probe.fwd.items(), key=lambda kv: -kv[1])[:12],
                "bwd_top": sorted(probe.bwd.items(), key=lambda kv: -kv[1])[:12],
                "bwd_order_head": probe.order[:5],
            }
            rows.append(row)
            log.info(json.dumps({k: row[k] for k in list(row)[:9]}))
    return rows


if __name__ == "__main__":
    main()
