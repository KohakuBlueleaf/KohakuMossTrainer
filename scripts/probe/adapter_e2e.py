"""Real-weight adapter round trip: a random LoRA on the live model vs the same LoRA
exported (comfy / lycoris) and loaded by MossPipeline (merge / bypass), fp32.

    python scripts/probe/adapter_e2e.py --weights <root> --embeds embeds.pt --out runs/e2e
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from kohakumoss.adapters import AdapterSpec, apply
from kohakumoss.inference import MossPipeline
from kohakumoss.inference.reference import rel_error
from kohakumoss.model import load_backbone
from kohakumoss.types.seqinfo import build_seqinfo

MULT = 0.8


def free():
    gc.collect()
    torch.cuda.empty_cache()


def probe_inputs(dev, text):
    g = torch.Generator().manual_seed(3)
    img = torch.randn((32 * 32, 128), generator=g).to(dev)
    seq = build_seqinfo(torch.tensor([[32, 32]]), torch.tensor([text.shape[0]]))
    return img, text.to(dev), torch.tensor([0.6], device=dev), seq.to(dev)


def run(model, dev, data):
    with torch.inference_mode():
        v = model(*probe_inputs(dev, data["embeds"][0])).float().cpu()
    res = MossPipeline.from_components(model, device=dev).generate(
        prompt_embeds=[data["embeds"][0].to(dev)],
        negative_embeds=[data["empty"].to(dev)],
        sizes=(512, 512),
        steps=20,
        seeds=[7],
        output="latent",
    )
    return v, res.latents[0].float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="root containing base/")
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    data = torch.load(args.embeds)
    base_dir = Path(args.weights) / "base"
    live, _ = load_backbone(base_dir, device=dev, dtype=torch.float32)
    live.requires_grad_(False)
    v_base, _ = run(live, dev, data)
    spec = AdapterSpec(algo="lora", preset="attn-mlp", dim=args.rank, alpha=args.rank)
    aset = apply(live, spec)
    torch.manual_seed(11)
    with torch.no_grad():
        for name, p in aset.named_trainable():
            if "lora_up" in name:
                p.normal_(0, 0.002)
    aset.set_multiplier(MULT)
    live.bind_ops()
    v_live, lat_live = run(live, dev, data)
    for fmt in ("comfy", "lycoris"):
        aset.export(str(out / f"{fmt}.safetensors"), fmt, dtype=None)
    aset.export(str(out / "comfy_bf16.safetensors"), "comfy")
    del live, aset
    free()
    report = {"adapter_effect_forward": rel_error(v_live, v_base), "runs": {}}
    for fmt in ("comfy", "lycoris", "comfy_bf16"):
        for mode in ("merge", "bypass"):
            pipe = MossPipeline.from_pretrained(
                args.weights,
                variant="base",
                dtype=torch.float32,
                device=dev,
                adapters=[(str(out / f"{fmt}.safetensors"), MULT)],
                adapter_mode=mode,
                text_encoder=False,
                vae=False,
            )
            v, lat = run(pipe.backbone, dev, data)
            row = {
                "forward_rel": rel_error(v, v_live),
                "latent_rel_20_steps": rel_error(lat, lat_live),
            }
            report["runs"][f"{fmt}/{mode}"] = row
            print(fmt, mode, row, flush=True)
            del pipe
            free()
    (out / "adapter_e2e.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
