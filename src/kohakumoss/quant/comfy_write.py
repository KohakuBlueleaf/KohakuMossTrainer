"""Write ComfyUI int8 ConvRot layers: for a layer `<n>`, `<n>.weight` (int8 codes of W R),
`<n>.weight_scale` (fp32 [N, 1]) and `<n>.comfy_quant` (`int8_tensorwise`, `convrot`,
`convrot_groupsize`), the layout `kohakumoss.quant.comfy_format` reads. Optional
per-block `layers.<i>.comfy_attention.config` runtime hints. See
docs/guides/export-to-comfy.md#int8-convrot-files.
"""

import json
import re

import torch
from torch import Tensor

from kohakumoss.quant.comfy_format import RECORD
from kohakumoss.quant.weights import quantize
from kohakumoss_kernels.hadamard import group_hadamard_eager

INT8_CONVROT = "int8_tensorwise"
BLOCK = re.compile(r"^layers\.(\d+)\.")
HINT_KEY = "layers.{}.comfy_attention.config"


def record_tensor(fields: dict) -> Tensor:
    """The uint8 tensor of a JSON record (`comfy_quant`, `comfy_attention.config`)."""
    data = json.dumps(fields).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


def convrot_int8(w: Tensor, group: int) -> tuple[Tensor, Tensor]:
    """(codes int8 [N, K], scale fp32 [N, 1]) of a 2-D weight: W R rotated in fp32
    (R = blockdiag of the normalised regular Hadamard of `group` over K), then per-row
    int8 with scale = max|row| / 127."""
    q, scale = quantize(group_hadamard_eager(w.float(), group), "int8")
    return q, scale[:, None]


def int8_convrot_state(
    state: dict[str, Tensor],
    layers: str,
    group: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    attention_hint: str = "",
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """(file state, {layer: relative L2 error of the dequantized weight}) of a bare-key
    DiT `state`: every 2-D `<n>.weight` whose module path fully matches the regex
    `layers` becomes an int8 ConvRot layer (rotated and quantized on `device`), every
    other floating tensor is cast to `dtype` (`rope.bands` stays fp32); a non-empty
    `attention_hint` adds each block's `comfy_attention.config` record."""
    pattern = re.compile(layers)
    rec = record_tensor(
        {"format": INT8_CONVROT, "convrot": True, "convrot_groupsize": group}
    )
    out, errors, blocks = {}, {}, set()
    for key, t in state.items():
        path = key[: -len(".weight")] if key.endswith(".weight") else ""
        if (m := BLOCK.match(key)) is not None:
            blocks.add(int(m.group(1)))
        if path and t.ndim == 2 and pattern.fullmatch(path):
            w = t.to(device=device, dtype=torch.float32)
            q, scale = convrot_int8(w, group)
            deq = group_hadamard_eager(q.float() * scale, group)
            errors[path] = ((deq - w).norm() / w.norm().clamp_min(1e-30)).item()
            out[key] = q.cpu()
            out[f"{path}.weight_scale"] = scale.cpu()
            out[f"{path}.{RECORD}"] = rec.clone()
            continue
        keep = key == "rope.bands" or not t.is_floating_point()
        out[key] = t if keep else t.to(dtype)
    if attention_hint:
        hint = record_tensor({"attention": attention_hint})
        for i in sorted(blocks):
            out[HINT_KEY.format(i)] = hint.clone()
    return out, errors
