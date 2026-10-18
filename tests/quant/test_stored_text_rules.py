"""Pre-quantized text-encoder file: the plan rules and the stored-layer key match
(model-free)."""

import pytest

from kohakumoss.encoders.text.plan import plan_problems, text_plan
from kohakumoss.quant.stored_text import _under

pytestmark = pytest.mark.cpu


def test_stored_needs_a_quantized_precision(tmp_path):
    f = tmp_path / "te.safetensors"
    f.write_bytes(b"")
    ok = plan_problems(text_plan(precision="fp8", stored=str(f)))
    assert not any("stored" in p for p in ok)
    bad = plan_problems(text_plan(precision="compute", stored=str(f)))
    assert any("stored" in p for p in bad)
    grouped = plan_problems(text_plan(precision="int8", quant_group=64, stored=str(f)))
    assert any("stored" in p for p in grouped)


def test_stored_file_must_exist(tmp_path):
    p = plan_problems(text_plan(precision="fp8", stored=str(tmp_path / "missing")))
    assert any("does not exist" in x for x in p)


def test_under_matches_only_stored_layers():
    layers = {"layers.0.mlp.up_proj"}
    assert _under("layers.0.mlp.up_proj.qweight.q", layers)
    assert _under("layers.0.mlp.up_proj.bias", layers)
    assert not _under("layers.0.mlp.up_proj_extra.weight", layers)
    assert not _under("layers.0.mlp.down_proj.weight", layers)
