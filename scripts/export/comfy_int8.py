"""Write a ComfyUI int8 ConvRot MOSS-Image transformer (the layout of the Comfy-Org
`moss_image_*_int8_convrot` files): the block linears matching LAYERS as int8 codes of
W R + per-row scales + a `comfy_quant` record, joint q/k/v fused, the rest as
SAVE_DTYPE, each block's attention hint. Any fold of WEIGHTS is undone first.

    kogine run scripts/export/comfy_int8.py --set WEIGHTS=merged_fp32.safetensors \
        --set OUT=moss_image_heal_int8_convrot.safetensors
"""

import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.io.save import export_state
from kohakumoss.quant.comfy_write import int8_convrot_state
from kohakumoss_common.knobs import check_script

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
log = get_logger("export")

WEIGHTS = ""  # HF transformer dir / root or a single file (any fold is undone)
LAYERS = r"layers\.\d+\.proj\.[a-z_]+"  # module paths (full regex match) written as int8 ConvRot
GROUP = 256  # ConvRot Hadamard group size (a power of 4 dividing every K)
SAVE_DTYPE = "bf16"  # the other tensors: "bf16" | "fp16"
ATTENTION_HINT = "comfy_kitchen_int8"  # each block's comfy_attention.config; "" = none
DEVICE = "cuda"  # where the weights are rotated and quantized
OUT = ""  # output .safetensors (report next to it as .json)


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not OUT:
        raise SystemExit("set WEIGHTS and OUT")
    start = time.perf_counter()
    model, load = load_backbone(WEIGHTS, device="cpu", dtype=torch.float32)
    state = export_state(model, dtype=None, fused_qkv=True)
    del model
    out, errors = int8_convrot_state(
        state, LAYERS, GROUP, DTYPES[SAVE_DTYPE], torch.device(DEVICE), ATTENTION_HINT
    )
    del state
    if not errors:
        raise SystemExit(f"LAYERS={LAYERS!r} matched no 2-D weight")
    save_file(out, str(OUT))
    rel = sorted(errors.values())
    report = {
        "out": str(OUT),
        "source": Path(WEIGHTS).name,
        "source_tensors": load.n_source,
        "group": GROUP,
        "layers": len(errors),
        "tensors": len(out),
        "save_dtype": SAVE_DTYPE,
        "attention_hint": ATTENTION_HINT,
        "rel_error": {"min": rel[0], "median": rel[len(rel) // 2], "max": rel[-1]},
        "rel_error_per_layer": errors,
        "bytes": Path(OUT).stat().st_size,
        "seconds": time.perf_counter() - start,
    }
    Path(OUT).with_suffix(".json").write_text(json.dumps(report, indent=1), "utf-8")
    log.info(
        json.dumps({k: v for k, v in report.items() if k != "rel_error_per_layer"})
    )


if __name__ == "__main__":
    main()
