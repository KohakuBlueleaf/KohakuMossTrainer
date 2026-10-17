"""The fp16-fix checkpoint: folded tensors + metadata, stock-eps load, exact undo."""

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from kohakumoss.model import load_backbone, random_backbone, save_single
from kohakumoss.model.io.keymap import FP16_FIX_KEY
from kohakumoss.model.numerics import fold, unfold
from tests.model.conftest import make_batch

pytestmark = pytest.mark.cpu


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def test_per_block_fold_refuses_keep_fold(tmp_path):
    model = random_backbone("tiny", seed=9)
    fold(model, [2.0**-2, 2.0**-5, 2.0**-3, 2.0**-6], "proj")
    with pytest.raises(ValueError):
        save_single(model, tmp_path / "x.safetensors", dtype=None, keep_fold=True)
    save_single(model, tmp_path / "plain.safetensors", dtype=None)


def test_fixed_file_roundtrip(tmp_path):
    base = random_backbone("tiny", seed=9)
    original = {k: v.clone() for k, v in base.state_dict().items()}
    save_single(base, tmp_path / "orig.safetensors", dtype=None)
    fold(base, 2.0**-3, "gate", (0.5, 1.0), eps="stock")
    save_single(base, tmp_path / "fix.safetensors", dtype=None, keep_fold=True)

    fixed = load_file(str(tmp_path / "fix.safetensors"))
    d = base.cfg.dim
    gate = fixed["shared_block_modulation.weight"][2 * d : 3 * d]
    assert torch.equal(
        gate, original["shared_block_modulation.weight"][2 * d : 3 * d] * 0.25
    )
    assert torch.equal(
        fixed["layers.1.proj.o.weight"], original["layers.1.proj.o.weight"] * 0.5
    )
    assert torch.equal(
        fixed["layers.1.proj.q.weight"], original["layers.1.proj.q.weight"]
    )
    with safe_open(str(tmp_path / "fix.safetensors"), framework="pt") as f:
        assert FP16_FIX_KEY in f.metadata()

    loaded, report = load_backbone(
        tmp_path / "fix.safetensors", device="cpu", dtype=torch.float32, keep_fp32=()
    )
    assert report.ok and loaded.fp16_fix == "weights" and loaded.fold_scales[0] == 0.125
    ref, _ = load_backbone(
        tmp_path / "orig.safetensors", device="cpu", dtype=torch.float32, keep_fp32=()
    )
    img, text, sigma, seq, _, _ = make_batch(base.cfg)
    assert _rel(loaded(img, text, sigma, seq), ref(img, text, sigma, seq)) < 1e-1
    unfold(loaded)
    for k, v in ref.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
