"""The weight fold scales in fp32: bit-exact for power-of-two factors in every dtype."""

import importlib

import pytest
import torch

from kohakumoss.model import random_backbone
from kohakumoss.model.numerics import fold, unfold
from kohakumoss.model.numerics.fold import _row_scales
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

FOLD_MODULE = importlib.import_module("kohakumoss.model.numerics.fold")


def _expected(state, rows, inverse=False):
    """fp64 reference of the scaled tensors, rounded once to their dtype."""
    out = {}
    for name, s in rows.items():
        t = state[name].double()
        if isinstance(s, torch.Tensor):
            s = s.double().reshape(-1, *([1] * (t.dim() - 1)))
        out[name] = (t / s if inverse else t * s).to(state[name].dtype)
    return out


def _check_fold(model, c, via, pregate=(1.0, 1.0)):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fold(model, c, via, pregate)
    rows = _row_scales(model, model.fold_scales, via, pregate)
    folded = {k: v.clone() for k, v in model.state_dict().items()}
    expected = _expected(before, rows)
    for name in rows:
        assert torch.equal(folded[name], expected[name]), name
    unfold(model)
    back = _expected(folded, rows, inverse=True)
    for name, t in model.state_dict().items():
        assert torch.equal(t, back.get(name, before[name])), name
    if next(model.parameters()).dtype != torch.float16:
        for name, t in model.state_dict().items():
            assert torch.equal(t, before[name]), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_power_of_two_fold_is_bit_exact(dtype):
    model = random_backbone("tiny", seed=4).to(dtype)
    _check_fold(model, 2.0**-4, "gate")
    _check_fold(model, 2.0**-4, "gate", pregate=(2.0**-1, 2.0**-2))
    depth = model.cfg.depth
    _check_fold(model, [2.0 ** -(2 + i % 3) for i in range(depth)], "proj")


def test_check_catches_lossy_scaling(monkeypatch):
    model = random_backbone("tiny", seed=4)

    def lossy(t, s, inverse):
        if isinstance(s, torch.Tensor):
            s = s.to(t.device, torch.bfloat16).reshape(-1, *([1] * (t.dim() - 1)))
        m = (1.0 / s) if inverse else s
        return (t.bfloat16() * m).to(t.dtype)

    monkeypatch.setattr(FOLD_MODULE, "_multiply", lossy)
    must_fail(lambda: _check_fold(model, 2.0**-4, "gate"), "bf16 scaling")
