"""ComfyUI quant records the loaders cannot run (model-free, constructed tensors): a
`comfy_quant` record of an unknown format is refused by `split_quantized` and by
`load_backbone` (before a weight is read), naming the layer; a ConvRot record splits
into a rotated base, and the layer without its record into a plain one (the checks
fail on them). The quant modules import first in a fresh interpreter."""

import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.model.io.load import load_backbone
from kohakumoss.quant.comfy import split_quantized
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CONVROT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
UNKNOWN = {"format": "nvfp4", "group_size": 16}
FIRST_IMPORTS = (
    "kohakumoss.quant.comfy",
    "kohakumoss.quant.comfy_format",
    "kohakumoss.quant.densify",
    "kohakumoss.quant.w4a8",
    "kohakumoss.encoders.text.comfy_load",
    "kohakumoss.training.exports",
)


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
    with pytest.raises(ValueError, match="blk.lin.comfy_quant.*nvfp4"):
        split_quantized(state)


def test_split_quantized_refuses_unknown_formats():
    check_refused(_state(UNKNOWN))
    must_fail(lambda: check_refused(_state(CONVROT)), "a ConvRot record")
    must_fail(lambda: check_refused(_state(None)), "the layer without its record")
    quant, dense = split_quantized(_state(CONVROT))
    assert quant["blk.lin"].rotation == 256 and dense == {}
    assert split_quantized(_state(None))[0]["blk.lin"].rotation == 0


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


@pytest.mark.budget(120)
def test_the_quant_modules_import_first():
    for module in FIRST_IMPORTS:
        check_imports_first(module)
    must_fail(
        lambda: check_imports_first("kohakumoss.quant.no_such_module"),
        "a module that does not import",
    )


def check_file_refused(path) -> None:
    """The load stops on the unknown record (a file it gets past fails later, on its
    missing model keys, which is not the refusal)."""
    try:
        load_backbone(path, device="cpu")
    except (ValueError, KeyError) as err:
        assert "ComfyUI quant format 'nvfp4'" in str(err), err
    else:
        raise AssertionError(f"{path} loaded")


def test_load_backbone_refuses_an_unknown_record(tmp_path):
    files = {}
    for name, record in (("unknown", UNKNOWN), ("convrot", CONVROT)):
        files[name] = tmp_path / f"{name}.safetensors"
        save_file(
            {
                "layers.0.proj.qkv.weight": torch.zeros(12, 256, dtype=torch.int8),
                "layers.0.proj.qkv.weight_scale": torch.ones(12, 1),
                "layers.0.proj.qkv.comfy_quant": _record(record),
            },
            str(files[name]),
        )
    check_file_refused(files["unknown"])
    must_fail(lambda: check_file_refused(files["convrot"]), "a ConvRot file")
