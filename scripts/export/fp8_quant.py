"""Write a pre-quantized fp8 / MXFP8 MOSS-Image transformer: every fp8-compute linear as
e4m3 `<n>.weight` (quantized from the checkpoint's stored values) + `<n>.weight_scale`,
the rest as SAVE_DTYPE, the recipe in the file metadata. `load_backbone` (and
TRANSFORMER=<file> in the sample and train scripts) attaches the stored bases without
building a 16-bit copy.

    kogine run scripts/export/fp8_quant.py --set WEIGHTS=<root>/base \
        --set PRECISION=mxfp8 --set OUT=moss_image_base_mxfp8.safetensors
"""

import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.build import DEFAULT_KEEP_FP32
from kohakumoss.model.io.load import base_keys
from kohakumoss.model.io.quant_source import backbone_source
from kohakumoss.quant.convert import default_filter, quantize_model
from kohakumoss.quant.stored import FP8_KEY, quantized_state
from kohakumoss_common.knobs import check_script

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
log = get_logger("export")

WEIGHTS = ""  # HF transformer dir / root or a single file
PRECISION = "mxfp8"  # "fp8_compute" | "mxfp8"
FP8_RECIPE = "row"  # fp8_compute scaling: "tensor" | "row" | "block"
SAVE_DTYPE = "bf16"  # the non-quantized tensors
MIN_FEATURES = 256  # linears smaller than this on either side stay dense
DEVICE = "cuda"  # where the weights are quantized
OUT = ""  # output .safetensors (report next to it as .json)


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not OUT:
        raise SystemExit("set WEIGHTS and OUT")
    if PRECISION not in ("fp8_compute", "mxfp8"):
        raise SystemExit(f"PRECISION={PRECISION!r}: 'fp8_compute' or 'mxfp8'")
    start = time.perf_counter()
    dtype = DTYPES[SAVE_DTYPE]
    model, load = load_backbone(WEIGHTS, device="cpu", dtype=dtype)
    accept = default_filter(DEFAULT_KEEP_FP32, MIN_FEATURES)
    source, scales = backbone_source(model)
    report = quantize_model(
        model,
        PRECISION,
        accept,
        dtype,
        FP8_RECIPE,
        work_device=torch.device(DEVICE),
        source=source,
        scales=scales,
    )
    state, meta = quantized_state(model, base_keys(model.cfg), dtype)
    meta["source"] = Path(WEIGHTS).name
    save_file(state, str(OUT), meta)
    out = {
        "out": str(OUT),
        "precision": PRECISION,
        "recipe": json.loads(meta[FP8_KEY])["recipe"],
        "quant": report,
        "source_tensors": load.n_source,
        "bytes": Path(OUT).stat().st_size,
        "seconds": time.perf_counter() - start,
    }
    Path(OUT).with_suffix(".json").write_text(json.dumps(out, indent=1), "utf-8")
    log.info(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
