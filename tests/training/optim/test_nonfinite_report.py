"""A non-finite step names its first non-finite gradients (parameter, adapter / base kind,
NaN and Inf counts; in-backward steps name the flagged parameters), and the instability
dump records them. A finite step reports nothing. Model-free: a base affine map plus
an adapter delta over explicit parameters, named as an adapted linear's."""

import json

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.precision import build_precision
from kohakumoss.training.callbacks.monitor import InstabilityDump
from kohakumoss.training.optim import build_bundle, build_param_groups
from kohakumoss.training.optim.bundle import base as bundle_module
from kohakumoss.training.state import TrainState
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _adapted(seed: int = 0) -> dict[str, torch.nn.Parameter]:
    """`base.weight` / `base.bias` (the base linear) and `base.lyco.weight` (its
    adapter's delta), 6 -> 5."""
    g = torch.Generator().manual_seed(seed)
    tensors = {
        "base.weight": torch.randn(5, 6, generator=g) / 6**0.5,
        "base.bias": torch.randn(5, generator=g) * 0.1,
        "base.lyco.weight": torch.randn(5, 6, generator=g) / 6**0.5,
    }
    return {n: torch.nn.Parameter(t) for n, t in tensors.items()}


def _forward(m: dict, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, m["base.weight"], m["base.bias"]) + F.linear(
        x, m["base.lyco.weight"]
    )


def _bundle(model: dict, fused: bool):
    groups = build_param_groups(list(model.items()), 1e-2, 0.0, "none")
    return build_bundle(
        groups,
        torch.device("cpu"),
        "adamw",
        {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8},
        {"mode": "constant", "end": -1},
        10,
        build_precision("fp16", ()),
        0.0,
        fused_step=fused,
    )


def _poisoned_step(fused: bool):
    model = _adapted()
    b = _bundle(model, fused)
    x = torch.randn(4, 6, generator=torch.Generator().manual_seed(1))
    b.backward(_forward(model, x).square().mean())
    model["base.lyco.weight"].grad.view(-1)[:3] = float("nan")
    model["base.bias"].grad[0] = float("inf")
    if fused:
        for p in model.values():
            b.step_param(p)
    b.step()
    return b


def _names_both(report: list[dict], counts: bool) -> None:
    by = {r["param"]: r for r in report}
    assert set(by) == {"base.lyco.weight", "base.bias"}, report
    assert by["base.lyco.weight"]["kind"] == "adapter"
    assert by["base.bias"]["kind"] == "base"
    if counts:
        assert (by["base.lyco.weight"]["nan"], by["base.lyco.weight"]["inf"]) == (3, 0)
        assert (by["base.bias"]["nan"], by["base.bias"]["inf"]) == (0, 1)


def test_plain_step_reports_counts(monkeypatch):
    _names_both(_poisoned_step(False).nonfinite_grads, counts=True)
    monkeypatch.setattr(bundle_module, "grad_report", lambda named: [])
    must_fail(
        lambda: _names_both(_poisoned_step(False).nonfinite_grads, True),
        "no grad report on a non-finite step",
    )


def test_fused_step_reports_flagged_params():
    _names_both(_poisoned_step(True).nonfinite_grads, counts=False)


def test_finite_step_reports_nothing():
    model = _adapted()
    b = _bundle(model, False)
    b.backward(_forward(model, torch.randn(4, 6)).square().mean())
    assert b.step() and b.nonfinite_grads == []


def test_instability_dump_records_the_report(tmp_path):
    b = _poisoned_step(False)
    dump = InstabilityDump(str(tmp_path), bundle=b)
    state = TrainState(step=7, loss=float("nan"), ok=False)

    class _Out:
        metrics: dict = {}

    dump.on_step_end(state, _Out())
    record = json.loads((tmp_path / "instability" / "step_00000007.json").read_text())
    _names_both(record["nonfinite_grads"], counts=True)
