"""Text-encoder bench on the real Qwen3.5 weights: tokens/s, host-to-device bytes and
peak VRAM per layout x prompts-per-call x memory mode x precision, plus the op choice
the registry selected by timing.

    python scripts/bench/textenc/textenc.py --weights <root>/base/text_encoder \
        --memory resident,offload,stream --layouts packed,padded,single \
        --batches 1,8,32 --prompts 64 --out runs/bench/textenc

One pass encodes --prompts captions (lengths drawn from --lengths); a row is the median
of --iters passes after one warm pass. Writes <out>/<gpu>_textenc.json and prints one
JSON row per configuration. A configuration that does not fit is recorded as OOM.
"""

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from kohakumoss.encoders.text import QwenTextEncoder, text_plan, tokenizer_dir
from kohakumoss_kernels.hardware.memory import device_used, window

acc = torch.accelerator

WORDS = (
    "1girl solo long hair smile outdoors cherry blossoms wind sky clouds city night "
    "neon rain reflections portrait masterpiece best quality detailed painting"
).split()


def captions(n: int, lengths: tuple[int, int], seed: int) -> list[str]:
    """`n` captions of uniformly drawn word counts (~1.3 tokens a word)."""
    rng = random.Random(seed)
    return [
        ", ".join(rng.choice(WORDS) for _ in range(rng.randint(*lengths)))
        for _ in range(n)
    ]


def run(args, tok, memory, layout, batch, precision, device) -> dict:
    gc.collect()
    acc.empty_cache()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}
    spec = {
        "memory": memory,
        "layout": layout,
        "batch": batch,
        "precision": precision,
        "dtype": args.dtype,
        "forward": args.forward,
        **json.loads(args.plan),
    }
    model = AutoModel.from_pretrained(args.weights, dtype=dtype[args.dtype])
    base = acc.memory_allocated(device)
    te = QwenTextEncoder(model, tok, device, text_plan(spec), output_device="cpu")
    held = acc.memory_allocated(device) - base
    prompts = captions(args.prompts, args.lengths, 0)
    te.encode(prompts)
    times, h2d = [], []
    with window(device) as w:
        for i in range(args.iters):
            before = te.placement.h2d_bytes
            t0 = time.perf_counter()
            te.encode(captions(args.prompts, args.lengths, i + 1))
            times.append(time.perf_counter() - t0)
            h2d.append(te.placement.h2d_bytes - before)
    tokens = te.stats["tokens"] / te.stats["passes"]
    med = sorted(times)[len(times) // 2]
    row = {
        "memory": te.plan.memory,
        "layout": layout,
        "batch": batch,
        "precision": precision,
        "dtype": args.dtype,
        "forward": args.forward,
        "prompts": args.prompts,
        "tokens_per_pass": tokens,
        "s_per_pass": med,
        "tokens_per_s": tokens / med,
        "h2d_GB_per_pass": sorted(h2d)[len(h2d) // 2] / 1e9,
        "peak_GiB": (w.alloc - base) / 2**30,
        "peak_reserved_GiB": w.reserved / 2**30,
        "device_used_GiB": device_used(device) / 2**30,
        "held_GiB": held / 2**30,
        "after_GiB": (acc.memory_allocated(device) - base) / 2**30,
        "quant": te.quant,
        "ops": te.op_names,
    }
    te.release()
    del te, model
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="text_encoder dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--memory", default="resident,offload,stream")
    ap.add_argument("--layouts", default="packed,padded,single")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--precisions", default="compute")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--forward", default="kernels")
    ap.add_argument("--plan", default="{}", help="more TextEncoderPlan fields (JSON)")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--lengths", default="20,160", help="min,max words per caption")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="runs/bench/textenc")
    args = ap.parse_args()
    args.lengths = tuple(int(x) for x in args.lengths.split(","))
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir(Path(args.weights))))
    rows = []
    for memory in args.memory.split(","):
        for layout in args.layouts.split(","):
            for batch in (int(b) for b in args.batches.split(",")):
                if layout == "single" and batch != 1:
                    continue
                for precision in args.precisions.split(","):
                    try:
                        row = run(args, tok, memory, layout, batch, precision, device)
                    except torch.OutOfMemoryError as err:
                        row = {"memory": memory, "layout": layout, "batch": batch,
                               "precision": precision, "error": "OOM",
                               "detail": str(err)[:200]}  # fmt: skip
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    gpu = getattr(torch, device.type).get_device_name(device).replace(" ", "_")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{gpu}_textenc_{args.dtype}_{args.forward}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"gpu": gpu, "args": vars(args), "rows": rows}, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
