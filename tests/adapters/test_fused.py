"""Fused LyCORIS paths bound by the applier, pinned and raced, against the eager body.

Both runs are judged against an fp64 eager reference with the same adapter tensors,
for the output and every adapter gradient: the fused result fails when it is clearly
worse than the eager one in both absolute and relative error (`_fused_bar`). Raced
runs also settle every slot on a precision-gated pick in a fresh race.
"""

from functools import partial

import pytest
import torch
from _adcases import make_spec
from _adtoy import inputs, perturb, toy
from _fused_bar import check
from lycoris.kernels import tier

from kohakumoss.adapters import apply
from kohakumoss.adapters.backend import pick_backend
from kohakumoss.adapters.bind.race import TierSlot
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
FUSED_CASES = (
    "lora",
    "loha",
    "lokr",
    "oft",
    "boft",
    "ia3",
    "glora",
    "dora-out",
    "doha",
)
UNRACED = ("ia3", "glora")


def _dtypes():
    if not torch.cuda.is_available():
        return [torch.float16]
    major, _ = torch.cuda.get_device_capability()
    return [torch.float16] if major < 8 else [torch.float16, torch.bfloat16]


def _run(name, mode, backend, dtype):
    model = toy().cuda().to(dtype)
    aset = apply(model, make_spec(name, mode=mode, backend=backend))
    perturb(aset)
    model.train()
    latent, text = (t.cuda().to(dtype) for t in inputs(n=64))
    y = model(latent, text)
    y.double().square().mean().backward()
    grads = {k: p.grad.double().clone() for k, p in aset.named_trainable()}
    return y.double().detach(), grads, aset


def _eager_runs(name, mode, dtype, patch):
    patch.setenv("LYCORIS_KERNEL_BACKEND", "torch")
    ref = _run(name, mode, "torch", torch.float64)
    eager = _run(name, mode, "torch", dtype)
    patch.delenv("LYCORIS_KERNEL_BACKEND")
    return ref, eager


def _slots(fn) -> list[TierSlot]:
    """Every TierSlot inside a bound forward (nested partials)."""
    if isinstance(fn, TierSlot):
        return [fn]
    if not isinstance(fn, partial):
        return []
    found = _slots(fn.func)
    for a in fn.args:
        found += _slots(a)
    return found


def _fresh_race(patch, tmp_path):
    """No tier decision known yet: this process races every op again (against a
    scratch table, so the shared one keeps its rows)."""
    patch.setenv("LYCORIS_KERNEL_CACHE_DIR", str(tmp_path))
    tier.reset("all")


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", FUSED_CASES)
def test_pinned_fused_is_as_accurate_as_eager(name, mode, dtype, monkeypatch):
    with monkeypatch.context() as patch:
        (y_r, g_r, _), (y_e, g_e, _) = _eager_runs(name, mode, dtype, patch)
    backend = pick_backend("cuda")
    y_f, g_f, aset = _run(name, mode, backend, dtype)
    assert aset.backend in ("triton", "tilelang")
    if name != "glora" or mode == "merge":
        assert any(t.fused for t in aset.targets), "no layer took the fused path"
    check(y_f, y_e, y_r, "output")
    for key in g_r:
        check(g_f[key], g_e[key], g_r[key], key)


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", [n for n in FUSED_CASES if n not in UNRACED])
def test_raced_layers_settle_on_gated_picks(name, mode, dtype, monkeypatch, tmp_path):
    with monkeypatch.context() as patch:
        (y_r, g_r, _), (y_e, g_e, _) = _eager_runs(name, mode, dtype, patch)
    with monkeypatch.context() as patch:
        _fresh_race(patch, tmp_path)
        y_f, g_f, aset = _run(name, mode, "auto", dtype)
        slots = [s for t in aset.targets for s in _slots(vars(t.linear)["forward"])]
        picks = tier.recorded()
        rejected, ungated = tier.rejected(), tier.ungated()
    assert slots, "no layer bound a TierSlot"
    assert all(s.arms for s in slots), "a slot never settled its pick"
    assert picks, "nothing was raced in this process"
    for key, pick in picks.items():
        assert pick not in rejected.get(key, {}), f"{key}: rejected pick {pick} kept"
    assert not ungated, f"raced without the precision gate: {sorted(ungated)[:2]}"
    check(y_f, y_e, y_r, "output")
    for key in g_r:
        check(g_f[key], g_e[key], g_r[key], key)


def test_the_bar_catches_a_wrong_kernel(monkeypatch):
    dtype = _dtypes()[0]
    with monkeypatch.context() as patch:
        (y_r, g_r, _), (y_e, g_e, _) = _eager_runs("lora", "bypass", dtype, patch)
    must_fail(lambda: check(y_e * 1.1, y_e, y_r, "output"), "a 10% output scale")
    key = next(iter(g_r))
    must_fail(lambda: check(g_e[key] * 1.1, g_e[key], g_r[key], key), "a 10% grad")
