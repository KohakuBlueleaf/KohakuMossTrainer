"""Dtype policy, fold row scales and saturation counters."""

import pytest
import torch

from kohakumoss.model import random_backbone
from kohakumoss.model.build import build_backbone, resolve_config
from kohakumoss.model.numerics import SaturationCounters, fold, output_row_scales
from tests.model.conftest import make_batch

pytestmark = pytest.mark.cpu


def test_keep_fp32_modules():
    model = build_backbone(resolve_config("tiny"), dtype=torch.bfloat16, init=True)
    assert model.shared_block_modulation.weight.dtype == torch.float32
    assert model.time_embedding[0].weight.dtype == torch.float32
    assert model.layers[0].proj["q"].weight.dtype == torch.bfloat16
    assert model.rope.bands.dtype == torch.float32


def test_output_row_scales_match_weight_change(tiny):
    before = {k: v.clone() for k, v in tiny.state_dict().items()}
    fold(tiny, 2.0**-3, "gate")
    scales = output_row_scales(tiny)
    for module, rows in scales.items():
        w0 = before[f"{module}.weight"]
        w1 = tiny.get_submodule(module).weight
        assert torch.equal(w1, w0 * rows.view(-1, *([1] * (w0.dim() - 1))).to(w0.dtype))
    d = tiny.cfg.dim
    gate = scales["shared_block_modulation"]
    assert torch.all(gate[2 * d : 3 * d] == 0.125) and torch.all(gate[:d] == 1.0)


def test_saturation_counters(tiny):
    img, text, sigma, seq, _, _ = make_batch(tiny.cfg)
    counters = SaturationCounters(tiny, threshold=0.0)
    out = counters.read()
    assert out == {}
    tiny(img, text, sigma, seq)
    stats = counters.read()
    assert set(stats) == {f"text.{i}" for i in range(2)} | {
        f"layer.{i}" for i in range(4)
    }
    assert all(v["over"] > 0 and v["maxabs"] > 0 for v in stats.values())
    counters.remove()
    tiny(img, text, sigma, seq)
    assert counters.read() == stats


def test_random_backbone_is_seeded():
    a = random_backbone("tiny", seed=4)
    b = random_backbone("tiny", seed=4)
    c = random_backbone("tiny", seed=5)
    wa, wb, wc = (m.layers[1].proj["o"].weight for m in (a, b, c))
    assert torch.equal(wa, wb) and not torch.equal(wa, wc)
