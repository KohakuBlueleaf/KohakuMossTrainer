"""fp16 magnitude probe: per-layer max |x| of every stream and pre-activation, fp32 model.

Records, over prompts x sigma x resolution: joint / text residual streams after each
block, attention logits (q.k * scale), ffn_up_gate and ffn_down outputs, o outputs, gates
and modulation. Writes a JSON report with the fold factor that puts the residual stream
under FOLD_TARGET.

    python scripts/probe/activations.py --weights <root>/base --embeds runs/probe/embeds.pt \
        --out runs/probe/activations.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from kohakumoss.model import load_backbone
from kohakumoss.types.seqinfo import build_seqinfo

SIGMAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.02)
RESOLUTIONS = ((512, 512), (1024, 1024), (768, 1344))
FOLD_TARGET = 2.0**12
FP16_MAX = 65504.0


class Recorder:
    """Running max per (site, layer) and per-call rows."""

    def __init__(self):
        self.peak = defaultdict(float)
        self.current = {}

    def put(self, site: str, layer: int, value: torch.Tensor) -> None:
        v = float(value.detach().abs().max())
        key = f"{site}.{layer}"
        self.current[key] = max(self.current.get(key, 0.0), v)
        self.peak[key] = max(self.peak[key], v)


def logit_max(q: torch.Tensor, k: torch.Tensor, cu: list[int]) -> torch.Tensor:
    """max |q k^T| * Dh^-0.5 over every head and segment."""
    scale = q.shape[-1] ** -0.5
    best = q.new_zeros(())
    for s, e in zip(cu[:-1], cu[1:]):
        for h in range(q.shape[1]):
            m = (q[s:e, h].float() @ k[s:e, h].float().T).abs().max()
            best = torch.maximum(best, m * scale)
    return best


def instrument(model, rec: Recorder, seq_box: dict):
    handles = []
    for i, block in enumerate(model.text_embedder.blocks):
        handles.append(block.register_forward_hook(_out_hook(rec, "text_stream", i)))
    for i, block in enumerate(model.layers):
        handles.append(block.register_forward_hook(_out_hook(rec, "stream", i)))
        for name in ("ffn_up_gate", "ffn_down", "o"):
            handles.append(
                block.proj[name].register_forward_hook(_out_hook(rec, name, i))
            )
        inner = block._qk

        def qk(*args, _inner=inner, _i=i):
            q, k = _inner(*args)
            cu = seq_box["seq"].attn_joint.cu_seqlens.tolist()
            rec.put("logit", _i, logit_max(q, k, cu))
            rec.put("q", _i, q)
            rec.put("k", _i, k)
            return q, k

        block._qk = qk
    return handles


def _out_hook(rec, site, layer):
    def hook(module, args, output):
        rec.put(site, layer, output)

    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--z0", default="", help="optional saved clean latents {(h,w): [1,C,h,w]}"
    )
    args = ap.parse_args()
    dev = torch.device(args.device)
    data = torch.load(args.embeds)
    texts = list(data["embeds"]) + [data["empty"]]
    names = list(data["prompts"]) + [""]
    z0_bank = torch.load(args.z0) if args.z0 else {}
    model, _ = load_backbone(
        args.weights, device=dev, dtype=torch.float32, keep_fp32=()
    )
    model.requires_grad_(False)
    rec, seq_box = Recorder(), {}
    instrument(model, rec, seq_box)
    rows = []
    with torch.inference_mode():
        for height, width in RESOLUTIONS:
            h, w = height // 16, width // 16
            for p, text in enumerate(texts):
                g = torch.Generator(dev).manual_seed(1234 + p)
                noise = torch.randn((h * w, 128), generator=g, device=dev)
                z0 = z0_bank.get((h, w))
                z0 = (
                    z0[0].flatten(1).T.to(dev)
                    if z0 is not None
                    else torch.randn((h * w, 128), generator=g, device=dev)
                )
                seq = build_seqinfo(
                    torch.tensor([[h, w]]), torch.tensor([text.shape[0]])
                )
                seq = seq.to(dev)
                seq_box["seq"] = seq
                for sigma in SIGMAS:
                    rec.current = {}
                    x = (1 - sigma) * z0 + sigma * noise
                    s = torch.tensor([sigma], device=dev)
                    mod, _ = model.conditioning(s)
                    out = model(x, text.to(dev), s, seq)
                    rows.append(
                        {
                            "res": [height, width],
                            "prompt": names[p][:40],
                            "sigma": sigma,
                            "mod_max": float(mod.abs().max()),
                            "gate_a_max": float(mod[:, 2].abs().max()),
                            "gate_f_max": float(mod[:, 5].abs().max()),
                            "out_max": float(out.abs().max()),
                            "stream_max": max(
                                v
                                for k, v in rec.current.items()
                                if k.startswith("stream.")
                            ),
                        }
                    )
                    print(rows[-1], flush=True)
    peak = dict(rec.peak)
    stream_peak = max(v for k, v in peak.items() if k.startswith("stream."))
    text_peak = max(v for k, v in peak.items() if k.startswith("text_stream."))
    k = max(0, math.ceil(math.log2(max(stream_peak, text_peak) / FOLD_TARGET)))
    sites = sorted({k_.split(".")[0] for k_ in peak})
    summary = {
        s: max(v for k_, v in peak.items() if k_.split(".")[0] == s) for s in sites
    }
    report = {
        "fold": 2.0**-k,
        "fold_target": FOLD_TARGET,
        "summary_max": summary,
        "over_fp16": {s: v for s, v in summary.items() if v > FP16_MAX},
        "per_layer": peak,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"fold": report["fold"], "summary_max": summary}, indent=1))


if __name__ == "__main__":
    main()
