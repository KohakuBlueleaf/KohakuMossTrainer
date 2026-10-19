"""ComfyUI quant records: a quantized layer `<n>` of a ComfyUI file carries
`<n>.comfy_quant`, a uint8 JSON record of its format. Layers this version cannot run
(a rotated `convrot` record, a format other than the plain int8 / fp8 ones) are
refused. Imports no model module (the checkpoint loader reads it)."""

import json
from collections.abc import Callable, Iterable

import torch
from torch import Tensor

RECORD = "comfy_quant"
PLAIN_FORMATS = ("int8_tensorwise", "float8_e4m3fn")


def read_record(t: Tensor) -> dict:
    """The JSON record of a `comfy_quant` uint8 tensor."""
    return json.loads(t.to(torch.uint8).numpy().tobytes().decode("utf-8"))


def unsupported_layers(
    keys: Iterable[str], get: Callable[[str], Tensor], formats=PLAIN_FORMATS
) -> list[str]:
    """`<path>: <record>` of every `comfy_quant` record among `keys` (read through
    `get`) that is rotated (`convrot`) or whose format is not in `formats`."""
    out = []
    for key in keys:
        if not key.endswith(f".{RECORD}"):
            continue
        record = read_record(get(key))
        if record.get("convrot") or record.get("format") not in formats:
            out.append(f"{key[: -len(RECORD) - 1]}: {json.dumps(record)}")
    return out


def refuse_comfy_quant(
    keys: Iterable[str],
    get: Callable[[str], Tensor],
    source: str,
    formats=PLAIN_FORMATS,
) -> None:
    """ValueError naming `source` when it holds ComfyUI quantized layers this version
    cannot run (`unsupported_layers`): loading them would compute with unrotated
    codes."""
    bad = unsupported_layers(keys, get, formats)
    if bad:
        raise ValueError(
            f"{source}: {len(bad)} ComfyUI quantized layers this version cannot run "
            f"(e.g. {bad[0]}); ConvRot int8 / W4A8 files (*_int8_convrot, *_w4a8_*) "
            "are not supported yet: use the bf16 ComfyUI file"
        )
