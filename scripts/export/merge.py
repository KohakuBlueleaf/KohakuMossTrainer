"""Merge adapter files of any supported format into a base transformer.

kogine run scripts/export/merge.py --config <file with WEIGHTS / ADAPTERS / OUT>
"""

from kohakumoss.adapters.export.cli import RENAMED, merge_main
from kohakumoss_common.knobs import check_script

WEIGHTS = ""  # base transformer (HF dir or single file)
ADAPTERS = []  # [(path, multiplier), ...]: LyCORIS, ComfyUI or training-state files
OUT = ""  # output file (or directory for FORMAT="diffusers")
FORMAT = "merged"  # "merged" | "diffusers"
SAVE_DTYPE = "bf16"  # dtype of the written tensors: "bf16" | "fp16" | "fp32"
FUSED_QKV = False
DEVICE = "cpu"


def main():
    check_script(globals(), RENAMED)
    return merge_main(globals())


if __name__ == "__main__":
    main()
