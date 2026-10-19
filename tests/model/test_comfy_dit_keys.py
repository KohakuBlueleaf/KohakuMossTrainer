"""The ComfyUI DiT key layout (model-free, constructed tensors): the fused `proj.qkv`
weight and its row scales split into q / k / v by rows with the quant record shared,
the runtime hints left out, and the header readers (stored precision, fused q/k/v)."""

import json

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.model.io.load import file_layers, stored_precision, stores_fused_qkv
from kohakumoss.model.io.source import CheckpointSource, ModelKeyView
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
RECORD = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def _bytes(payload: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)


def _file(tmp_path, quantized: bool = True) -> tuple:
    g = torch.Generator().manual_seed(0)
    qkv = torch.randint(-127, 128, (12, 256), generator=g, dtype=torch.int8)
    scale = torch.rand(12, 1, generator=g)
    state = {
        "layers.0.proj.qkv.weight": qkv if quantized else qkv.bfloat16(),
        "layers.0.proj.o.weight": torch.randn(4, 256).bfloat16(),
        "layers.0.norm1.weight": torch.ones(4).bfloat16(),
        "layers.0.comfy_attention.config": _bytes({"attention": "comfy_kitchen_int8"}),
    }
    if quantized:
        state["layers.0.proj.qkv.weight_scale"] = scale
        state["layers.0.proj.qkv.comfy_quant"] = _bytes(RECORD)
    path = tmp_path / ("int8.safetensors" if quantized else "bf16.safetensors")
    save_file(state, str(path))
    return path, qkv, scale


def check_split(view, qkv, scale) -> None:
    for i, part in enumerate("qkv"):
        assert torch.equal(
            view.get(f"layers.0.proj.{part}.weight"), qkv[4 * i : 4 * i + 4]
        )
        got = view.get(f"layers.0.proj.{part}.weight_scale")
        assert torch.equal(got, scale[4 * i : 4 * i + 4])
        assert view.shape(f"layers.0.proj.{part}.weight") == (4, 256)


def test_fused_qkv_splits_weight_and_scales_by_rows(tmp_path):
    path, qkv, scale = _file(tmp_path)
    view = ModelKeyView(CheckpointSource(path))
    check_split(view, qkv, scale)
    assert view.hints == 1
    assert not any("comfy_attention" in key for key in view.keys())
    record = view.get("layers.0.proj.k.comfy_quant")
    assert json.loads(record.numpy().tobytes()) == RECORD
    layers = file_layers(view, None)
    assert {p: lay.rotation for p, lay in layers.items()} == {
        f"layers.0.proj.{p}": 256 for p in "qkv"
    }
    assert file_layers(view, {"recipe": "row", "layers": []}) == {}
    must_fail(lambda: check_split(view, qkv, scale.flip(0)), "scale rows reversed")


def test_header_readers(tmp_path):
    int8, _, _ = _file(tmp_path)
    bf16, _, _ = _file(tmp_path, quantized=False)
    assert stored_precision(int8) == "int8"
    assert stored_precision(bf16) == "compute"
    assert stores_fused_qkv(int8) and stores_fused_qkv(bf16)
    split = tmp_path / "split.safetensors"
    save_file({"layers.0.proj.q.weight": torch.zeros(4, 4)}, str(split))
    assert not stores_fused_qkv(split)
    must_fail(lambda: _assert_int8(bf16), "a 16-bit file")


def _assert_int8(path) -> None:
    assert stored_precision(path) == "int8"
