"""Merge adapter files of any supported format into a base transformer.

kogine run scripts/export/merge.py --config <file with WEIGHTS / ADAPTERS / OUT>
"""

from kohakumoss.adapters.cli import merge_main

WEIGHTS = ""  # base transformer (HF dir or single file)
ADAPTERS = []  # [(path, multiplier), ...]: LyCORIS, ComfyUI or training-state files
OUT = ""  # output file (or directory for FORMAT="diffusers")
FORMAT = "merged"  # "merged" | "diffusers"
DTYPE = "bf16"  # "bf16" | "fp16" | "fp32"
FUSED_QKV = False
DEVICE = "cpu"

if __name__ == "__main__":
    merge_main(globals())
