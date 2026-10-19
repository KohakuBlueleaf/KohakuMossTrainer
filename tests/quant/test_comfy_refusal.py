"""ComfyUI quantized layers this version cannot run (model-free, constructed tensors):
a `comfy_quant` record that is rotated (ConvRot) or names another format is refused by
`split_quantized`, naming the layer, and a DiT file holding any record is refused by
`load_backbone` before a weight is read; the same layers without the record load as
before (the check fails on them)."""

import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.model.io.load import load_backbone
from kohakumoss.quant.comfy import split_quantized
from kohakumoss.quant.comfy_format import unsupported_layers
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CONVROT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def _record(payload: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)


def _state(record: dict | None) -> dict:
    g = torch.Generator().manual_seed(0)
    state = {
        "blk.lin.weight": torch.randint(-127, 128, (32, 256), generator=g).to(
            torch.int8
        ),
        "blk.lin.weight_scale": torch.rand(32, 1, generator=g),
    }
    if record is not None:
        state["blk.lin.comfy_quant"] = _record(record)
    return state


def check_refused(state: dict) -> None:
    with pytest.raises(ValueError, match="blk.lin.*ConvRot"):
        split_quantized(state)


@pytest.mark.parametrize(
    "record", (CONVROT, {"format": "asym_w4a8_int8", "group_size": 16})
)
def test_split_quantized_refuses_rotated_and_unknown_formats(record):
    check_refused(_state(record))
    must_fail(lambda: check_refused(_state(None)), "the layer without its record")
    plain = _state({"format": "int8_tensorwise"})
    quant, dense = split_quantized(plain)
    assert set(quant) == {"blk.lin"} and dense == {}
    assert unsupported_layers(plain, plain.__getitem__) == []


def check_imports_first(module: str) -> None:
    """`module` imports first in a fresh interpreter (no import cycle through the
    checkpoint loader)."""
    out = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-1500:]


@pytest.mark.budget(60)
def test_the_quant_modules_import_first():
    for module in ("kohakumoss.quant.comfy", "kohakumoss.quant.comfy_format"):
        check_imports_first(module)
    must_fail(
        lambda: check_imports_first("kohakumoss.quant.no_such_module"),
        "a module that does not import",
    )


def check_file_refused(path) -> None:
    """The load stops with the refusal (a file it gets past fails later, on its
    missing model keys, which is not the refusal)."""
    try:
        load_backbone(path, device="cpu")
    except (ValueError, KeyError) as err:
        assert "ComfyUI quantized layers" in str(err), err
    else:
        raise AssertionError(f"{path} loaded")


def test_load_backbone_refuses_a_comfy_quantized_file(tmp_path):
    quantized = tmp_path / "int8_convrot.safetensors"
    save_file(
        {
            "layers.0.proj.qkv.weight": torch.zeros(12, 256, dtype=torch.int8),
            "layers.0.proj.qkv.weight_scale": torch.ones(12, 1),
            "layers.0.proj.qkv.comfy_quant": _record(CONVROT),
        },
        str(quantized),
    )
    check_file_refused(quantized)
    plain = tmp_path / "plain.safetensors"
    save_file({"layers.0.proj.qkv.weight": torch.zeros(12, 256)}, str(plain))
    must_fail(lambda: check_file_refused(plain), "a file without quant records")
