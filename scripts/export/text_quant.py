"""Write a pre-quantized MOSS-Image text encoder: every quantized linear as
`<n>.weight` (codes, quantized from the checkpoint's stored values, so they equal
quantize-at-load at any compute dtype) + `<n>.weight_scale` (+ `<n>.bias`), the rest
as SAVE_DTYPE, the precision / recipe / layers in the file metadata. Training and
inference load it with the text-encoder plan field `stored` (TEXT_ENCODE), so the
encoder is not quantized at every load; the config and tokenizer stay in WEIGHTS.

    kogine run scripts/export/text_quant.py --set WEIGHTS=<root>/base \
        --set PRECISION=fp8_compute --set OUT=moss_text_encoder_fp8.safetensors
"""

import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel

from kohakumoss.logging.log import get_logger, setup
from kohakumoss.quant.source import text_source
from kohakumoss.quant.stored_text import TEXT_QUANT_KEY, quantized_text_state, text_of
from kohakumoss.quant.text import quantize_text_linears
from kohakumoss_common.knobs import check_script

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
log = get_logger("export")

WEIGHTS = ""  # pipeline root (text_encoder/) or the text-encoder dir
PRECISION = "fp8_compute"  # "int8" | "fp8" (weight-only) | "fp8_compute" | "mxfp8"
FP8_RECIPE = "row"  # fp8_compute scaling: "tensor" | "row" | "block"
SAVE_DTYPE = "bf16"  # the non-quantized tensors and the compute dtype
MIN_FEATURES = 256  # linears smaller than this on either side stay dense
DEVICE = "cuda"  # where the weights are quantized
OUT = ""  # output .safetensors (report next to it as .json)


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not OUT:
        raise SystemExit("set WEIGHTS and OUT")
    if PRECISION not in ("int8", "fp8", "fp8_compute", "mxfp8"):
        raise SystemExit(f"PRECISION={PRECISION!r}: int8 / fp8 / fp8_compute / mxfp8")
    start = time.perf_counter()
    root = Path(WEIGHTS)
    model_dir = root / "text_encoder" if (root / "text_encoder").is_dir() else root
    dtype = DTYPES[SAVE_DTYPE]
    model = AutoModel.from_pretrained(str(model_dir), dtype=dtype)
    text = text_of(model)
    report = quantize_text_linears(
        text,
        PRECISION,
        0,
        dtype,
        MIN_FEATURES,
        FP8_RECIPE,
        work_device=torch.device(DEVICE),
        source=text_source(model_dir),
    )
    state, meta = quantized_text_state(text, PRECISION, dtype)
    meta["source"] = model_dir.name
    save_file(state, str(OUT), meta)
    out = {
        "out": str(OUT),
        "precision": PRECISION,
        "recipe": json.loads(meta[TEXT_QUANT_KEY])["recipe"],
        "quant": report,
        "bytes": Path(OUT).stat().st_size,
        "seconds": time.perf_counter() - start,
    }
    Path(OUT).with_suffix(".json").write_text(json.dumps(out, indent=1), "utf-8")
    log.info(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
