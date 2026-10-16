"""Encode the probe prompts (and the empty prompt) once; later probes load the file.

python scripts/probe/embed.py --weights <root>/base --out runs/probe/embeds.pt
"""

import argparse
from pathlib import Path

import torch

from kohakumoss.encoders import QwenTextEncoder
from kohakumoss.inference.reference import PROBE_PROMPTS as PROMPTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    te = QwenTextEncoder.from_pretrained(args.weights, args.device, torch.bfloat16)
    embeds = [e.float().cpu() for e in te.encode(list(PROMPTS))]
    empty = te.empty().float().cpu()
    ids = [te.token_ids(p)[0].tolist() for p in PROMPTS]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"prompts": list(PROMPTS), "embeds": embeds, "empty": empty, "ids": ids},
        args.out,
    )
    print({"lengths": [e.shape[0] for e in embeds], "empty": empty.shape[0]})


if __name__ == "__main__":
    main()
