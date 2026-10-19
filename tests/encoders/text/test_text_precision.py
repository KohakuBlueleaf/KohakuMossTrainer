"""Text-encoder precision `auto` (model-free): it takes the format the weights store
(compute on the host), a stored-only precision on other weights is refused, the W4A8
decode knob is checked, the cache spec names a stored layout, and a ComfyUI file's
stored format is read from its headers and records (constructed tensors)."""

import json

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.encoders.specs import text_spec
from kohakumoss.encoders.text.comfy_load import text_file_format
from kohakumoss.encoders.text.plan import plan_problems, resolve_precision, text_plan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
INT8 = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
W4A8 = {
    "format": "asym_w4a8_int8",
    "group_size": 16,
    "convrot": True,
    "convrot_groupsize": 256,
}


def test_auto_takes_the_stored_format():
    auto = text_plan({})
    assert auto.precision == "auto" and auto.w4a8 == "auto"
    assert resolve_precision(auto, "int8", "cuda").precision == "int8"
    assert resolve_precision(auto, "compute", "cuda").precision == "compute"
    assert resolve_precision(auto, "w4a8", "cpu").precision == "compute"
    on_host = text_plan({"memory": "cpu"})
    assert resolve_precision(on_host, "int8", "cuda").precision == "compute"
    explicit = text_plan({"precision": "fp8"})
    assert resolve_precision(explicit, "int8", "cuda").precision == "fp8"
    with pytest.raises(ValueError, match="stored format"):
        resolve_precision(text_plan({"precision": "w4a8"}), "int8", "cuda")
    must_fail(
        lambda: _assert_precision(auto, "int8", "cpu", "int8"), "int8 on the host"
    )


def _assert_precision(plan, stored, device, want) -> None:
    assert resolve_precision(plan, stored, device).precision == want


def test_w4a8_knob_and_spec():
    assert plan_problems(text_plan({"w4a8": "packed"})) == []
    assert any("w4a8" in p for p in plan_problems(text_plan({"w4a8": "int4"})))
    plan = text_plan({"precision": "int8"})
    assert text_spec(plan, torch.bfloat16) == "bf16/int8/g0"
    stored = text_spec(plan, torch.bfloat16, "int8-convrot256")
    assert stored == "bf16/int8/g0/src:int8-convrot256"
    must_fail(lambda: _assert_same(plan), "the stored layout dropped")


def _assert_same(plan) -> None:
    assert text_spec(plan, torch.bfloat16, "int8-convrot256") == text_spec(
        plan, torch.bfloat16
    )


def _record(payload: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)


def _te_file(tmp_path, linear: dict | None) -> str:
    prefix = "model.language_model."
    state = {
        f"{prefix}embed_tokens.weight": torch.zeros(8, 256, dtype=torch.int8),
        f"{prefix}embed_tokens.weight_scale": torch.ones(8, 1),
        f"{prefix}embed_tokens.comfy_quant": _record(INT8),
        f"{prefix}norm.weight": torch.ones(256).bfloat16(),
        "lm_head.weight": torch.zeros(8, 256, dtype=torch.int8),
        "lm_head.weight_scale": torch.ones(8, 1),
        "lm_head.comfy_quant": _record(INT8),
    }
    proj = f"{prefix}layers.0.mlp.up_proj"
    if linear is None:
        state[f"{proj}.weight"] = torch.zeros(32, 256).bfloat16()
    elif linear["format"] == "asym_w4a8_int8":
        state[f"{proj}.weight"] = torch.zeros(32, 128, dtype=torch.int8)
        state[f"{proj}.weight_s_rel"] = torch.ones(32, 16).to(torch.float8_e4m3fn)
        state[f"{proj}.weight_s_channel"] = torch.ones(32)
        state[f"{proj}.weight_codebook"] = torch.zeros(16)
        state[f"{proj}.comfy_quant"] = _record(linear)
    else:
        state[f"{proj}.weight"] = torch.zeros(32, 256, dtype=torch.int8)
        state[f"{proj}.weight_scale"] = torch.ones(32, 1)
        state[f"{proj}.comfy_quant"] = _record(linear)
    path = tmp_path / f"te_{len(state)}_{linear and linear['format']}.safetensors"
    save_file(state, str(path))
    return str(path)


def test_comfy_text_file_formats(tmp_path):
    assert text_file_format(_te_file(tmp_path, INT8)) == ("int8", "int8-convrot256")
    assert text_file_format(_te_file(tmp_path, W4A8)) == (
        "w4a8",
        "int8-convrot256+w4a8-g16-convrot256",
    )
    assert text_file_format(_te_file(tmp_path, None)) == ("compute", "")
    must_fail(
        lambda: _assert_format(_te_file(tmp_path, None), "int8"),
        "a file with a 16-bit linear and an int8 embedding",
    )


def _assert_format(path, want) -> None:
    assert text_file_format(path)[0] == want
