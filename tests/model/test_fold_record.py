"""The weight-fold record a training state carries: it rebuilds the fold's row scales on
any model of the architecture, and folds compare by what they scale. The backbone is the real architecture on the meta device (shapes
only; the scales are host tensors)."""

import pytest
import torch

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.numerics import (
    fold,
    fold_record,
    output_row_scales,
    record_row_scales,
    same_fold,
)
from kohakumoss.types.arch import PRESETS
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
FIX = {"c": 1 / 16, "s_attn": 0.5, "s_ffn": 0.25}


def meta_model() -> MossBackbone:
    return MossBackbone(PRESETS["base"], device="meta")


def folded() -> MossBackbone:
    """The fold FP16_FIX='weights' applies with FIX (the uniform gate fold)."""
    model = meta_model()
    fold(model, FIX["c"], "gate", (FIX["s_attn"], FIX["s_ffn"]))
    return model


def assert_same_scales(a: dict, b: dict) -> None:
    assert sorted(a) == sorted(b)
    for path in a:
        assert torch.equal(a[path], b[path]), path


def test_record_rebuilds_the_row_scales_on_an_unfolded_model():
    model = folded()
    record = fold_record(model)
    assert record is not None and fold_record(meta_model()) is None
    rebuilt = record_row_scales(meta_model(), record)
    assert_same_scales(rebuilt, output_row_scales(model))
    dim = PRESETS["base"].dim
    assert rebuilt["layers.0.proj.o"][0] == 0.5
    assert rebuilt["shared_block_modulation"][5 * dim] == (1 / 16) / 0.25
    wrong = dict(record, pregate=[1.0, 1.0])
    must_fail(
        lambda: assert_same_scales(record_row_scales(meta_model(), wrong), rebuilt),
        "a record without the pre-gate scales",
    )
    assert record_row_scales(meta_model(), None) == {}


def _scales_check(scales: dict) -> None:
    dim = PRESETS["base"].dim
    o = scales["layers.3.proj.o"]
    assert o.shape == (dim,) and bool((o == 0.5).all())


def test_row_scales_of_a_stored_base():
    """A quantized / streamed layer's dense weight is an empty placeholder: the fold's row
    scales still cover every output row (the width of the layer, not of the weight)."""
    model = folded()
    _scales_check(output_row_scales(model))
    layer = model.get_submodule("layers.3.proj.o")
    layer.weight = torch.nn.Parameter(
        torch.empty(0, device="meta"), requires_grad=False
    )
    _scales_check(output_row_scales(model))
    n = PRESETS["base"].dim
    must_fail(
        lambda: _scales_check({"layers.3.proj.o": torch.full((0,), 0.5)}),
        "placeholder width",
    )
    assert layer.out_features == n


def _same(a, b) -> None:
    assert same_fold(a, b)


def test_same_fold_compares_what_is_scaled():
    uniform = {"scales": [0.0625] * 42, "via": "gate", "pregate": [1.0, 1.0]}
    one = {"scales": [0.0625], "via": "gate", "pregate": [1, 1], "eps": "stock"}
    assert same_fold(uniform, one)
    assert same_fold(None, None) and not same_fold(None, one)
    must_fail(lambda: _same(one, {**one, "via": "proj"}), "a per-block fold")
    must_fail(lambda: _same(one, {**one, "scales": [0.125]}), "another factor")
