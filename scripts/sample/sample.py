"""Generate images with MossPipeline; writes PNGs plus metadata.json of every setting.

    python scripts/sample/sample.py --weights <root> --prompt "a cat" --size 1024x1024 \
        --seed 0 --out out/samples
    python scripts/sample/sample.py --weights <root> --prompts-file prompts.json \
        --adapter my_lora.safetensors:0.8 --dtype fp16 --fp16-fix online \
        --fp16-fix-kwargs '{"c": 0.0625}' --residency offload

A prompts file is a .txt (one prompt per line) or a .json list of strings or of objects
{"prompt", "negative", "size": [H, W], "seed"}.
"""

import argparse
import json
from pathlib import Path

import torch

from kohakumoss.inference import MossPipeline

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_size(text: str) -> tuple[int, int]:
    h, w = text.lower().split("x")
    return int(h), int(w)


def parse_adapter(text: str) -> tuple[str, float]:
    path, _, mult = text.rpartition(":")
    if not path or not mult.replace(".", "", 1).lstrip("-").isdigit():
        return text, 1.0
    return path, float(mult)


def load_items(args) -> list[dict]:
    items = [{"prompt": p} for p in args.prompt]
    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix == ".json":
            for entry in json.loads(path.read_text(encoding="utf-8")):
                items.append(entry if isinstance(entry, dict) else {"prompt": entry})
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
            items += [{"prompt": line} for line in lines if line.strip()]
    if not items:
        raise SystemExit("no prompts: pass --prompt or --prompts-file")
    sizes = [parse_size(s) for s in args.size] or [(1024, 1024)]
    for i, item in enumerate(items):
        item.setdefault("negative", args.negative)
        item["size"] = tuple(item.get("size") or sizes[min(i, len(sizes) - 1)])
        seed = args.seed[i] if i < len(args.seed) else args.seed_base + i
        item.setdefault("seed", seed)
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="weights root or pipeline dir")
    ap.add_argument("--variant", default=None, help="base | sft subdir of the root")
    ap.add_argument("--transformer", default=None, help="DiT override (file / dir)")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--prompts-file", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument(
        "--size", action="append", default=[], help="HxW, per prompt or one"
    )
    ap.add_argument("--seed", type=int, action="append", default=[])
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument(
        "--skip-layers", default="default", help="'default', 'none' or 22,23"
    )
    ap.add_argument("--cfg-mode", default="packed", choices=("packed", "two_pass"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=tuple(DTYPES))
    ap.add_argument(
        "--fp16-fix",
        default="none",
        choices=("none", "weights", "online", "fp32_residual"),
    )
    ap.add_argument(
        "--fp16-fix-kwargs",
        default="{}",
        help='JSON, e.g. {"c": 0.0625, "clamp": true}',
    )
    ap.add_argument("--adapter", action="append", default=[], help="path[:multiplier]")
    ap.add_argument("--adapter-mode", default="merge", choices=("merge", "bypass"))
    ap.add_argument("--residency", default="gpu", choices=("gpu", "offload", "te_once"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--tune", default="auto", help="kernel tuning: auto | off | prewarm"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    match args.skip_layers:
        case "default":
            skip = None
        case "none":
            skip = ()
        case text:
            skip = tuple(int(v) for v in text.split(","))
    items = load_items(args)
    pipe = MossPipeline.from_pretrained(
        args.weights,
        variant=args.variant,
        transformer=args.transformer,
        dtype=DTYPES[args.dtype],
        device=args.device,
        adapters=[parse_adapter(a) for a in args.adapter],
        adapter_mode=args.adapter_mode,
        residency=args.residency,
        fp16_fix=args.fp16_fix,
        fp16_fix_kwargs=json.loads(args.fp16_fix_kwargs),
        tune=args.tune,
    )
    result = pipe.generate(
        [it["prompt"] for it in items],
        negative_prompts=[it["negative"] for it in items],
        sizes=[it["size"] for it in items],
        steps=args.steps,
        cfg=args.cfg,
        shift=args.shift,
        skip_layers=skip,
        seeds=[it["seed"] for it in items],
        batch=args.batch,
        cfg_mode=args.cfg_mode,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (image, item) in enumerate(zip(result.images, items)):
        name = f"{i:04d}_{item['seed']}.png"
        image.save(out / name)
        files.append({"file": name, **item, "size": list(item["size"])})
    meta = {
        "cli": vars(args),
        "settings": result.settings,
        "images": files,
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=1, default=str), encoding="utf-8"
    )
    print(f"wrote {len(files)} images to {out}")


if __name__ == "__main__":
    main()
