"""The fp16 fixes on the real backbone and two encoded dataset samples. FP16_FIX keeps
one consistent state (idempotent, conflict-checked, cleared exactly); the weight fold
scales exactly the rows it names and its undo restores every tensor bit for bit (bf16);
in fp16 the online fold, the in-memory weight fold and the fp16-fix file each match the
fp32-residual run at least as closely as the bf16 model does."""

import gc
import importlib
from functools import partial

import pytest
import torch
from torch import Tensor

from kohakumoss.model import load_backbone
from kohakumoss.model.io.load import read_fp16_fix
from kohakumoss.model.numerics import apply_fp16_fix, clear_fp16_fix
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
FOLD_MODULE = importlib.import_module("kohakumoss.model.numerics.fold")
C = 2.0**-4
BASE_BF16 = ("base", torch.bfloat16)
BASE_FP16 = ("base", torch.float16)
FILE_FP16 = ("file", torch.float16)
FIXES = {"fp32_residual": {}, "online": {"c": C}, "weights": {"c": C}}


class OneModel:
    """At most one backbone on the device: `get(key)` frees the held one first."""

    def __init__(self, load):
        self.load = load
        self.key = None
        self.model = None

    def get(self, key):
        if self.key != key:
            self.free()
            self.model = self.load(key)
            self.key = key
        return self.model

    def free(self) -> None:
        self.model = self.key = None
        gc.collect()
        torch.cuda.empty_cache()


def _load(weights_root, fix_file, key):
    source, dtype = key
    if source == "file" and fix_file is None:
        pytest.skip("no fp16-fix file")
    path = weights_root / "base" if source == "base" else fix_file
    model, report = load_backbone(path, device="cuda", dtype=dtype)
    assert report.ok
    return model.requires_grad_(False)


@pytest.fixture(scope="module")
def models(weights_root, fix_file):
    held = OneModel(partial(_load, weights_root, fix_file))
    yield held
    held.free()


@torch.inference_mode()
def forward(model, batch) -> Tensor:
    return model(*batch).float().cpu()


def _rel(a: Tensor, b: Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def check_within(out: Tensor, ref: Tensor, ruler: Tensor) -> None:
    """`out` is finite and no farther from `ref` than the bf16 run `ruler` is."""
    assert torch.isfinite(ref).all() and ref.abs().max() > 0, "reference unusable"
    assert torch.isfinite(out).all(), "non-finite output"
    err, floor = _rel(out, ref), _rel(ruler, ref)
    assert err <= floor, f"rel error {err:.3e} > the bf16 run's {floor:.3e}"


def _eps(model) -> dict[str, float]:
    return {n: m.eps for n, m in model.named_modules() if hasattr(m, "eps")}


def _fold_rows(model) -> dict:
    """Parameter name -> multiplier of the uniform gate fold by C."""
    return FOLD_MODULE._row_scales(model, (C,) * model.cfg.depth, "gate")


def _snapshot(model, names) -> dict[str, Tensor]:
    params = dict(model.named_parameters())
    return {n: params[n].detach().cpu().clone() for n in names}


def _scaled(t: Tensor, s) -> Tensor:
    """fp64 reference of `t * s` (per-row `s` broadcast), rounded once to t's dtype."""
    x = t.double()
    if isinstance(s, Tensor):
        s = s.double().reshape(-1, *([1] * (x.dim() - 1)))
    return (x * s).to(t.dtype)


def check_folded(model, before: dict[str, Tensor]) -> None:
    """Every folded tensor equals its pre-fold value times its fold multiplier."""
    params = dict(model.named_parameters())
    for name, s in _fold_rows(model).items():
        assert torch.equal(params[name].cpu(), _scaled(before[name], s)), name


def check_restored(model, before: dict[str, Tensor]) -> None:
    params = dict(model.named_parameters())
    for name, t in before.items():
        assert torch.equal(params[name].cpu(), t), name


@torch.no_grad()
def _put_back(model, before: dict[str, Tensor]) -> None:
    params = dict(model.named_parameters())
    for name, t in before.items():
        params[name].copy_(t)


def test_fix_state_is_consistent(models):
    model = models.get(BASE_BF16)
    eps = _eps(model)
    apply_fp16_fix(model, "online", c=C)
    online = _eps(model)
    assert online != eps and model.fp16_fix == "online"
    apply_fp16_fix(model, "online", c=C)
    assert _eps(model) == online, "the same fix twice compounded eps"
    with pytest.raises(ValueError):
        apply_fp16_fix(model, "online", c=2 * C)
    with pytest.raises(ValueError):
        apply_fp16_fix(model, "weights", c=C)
    clear_fp16_fix(model)
    assert _eps(model) == eps and model.fp16_fix == "none"
    must_fail(lambda: _same(online, eps), "eps of the online fix read as stock eps")


def _same(a, b) -> None:
    assert a == b


def test_weight_fold_round_trips_exactly(models, monkeypatch):
    model = models.get(BASE_BF16)
    before = _snapshot(model, _fold_rows(model))
    apply_fp16_fix(model, "weights", c=C)
    assert model.fp16_fix == "weights" and model.fold_scales[0] == C
    check_folded(model, before)
    clear_fp16_fix(model)
    check_restored(model, before)
    must_fail(lambda: check_folded(model, before), "an unfolded model read as folded")
    try:
        with monkeypatch.context() as mp:
            mp.setattr(FOLD_MODULE, "unscaled", lambda t, s: t)
            apply_fp16_fix(model, "weights", c=C)
            clear_fp16_fix(model)
        must_fail(lambda: check_restored(model, before), "an undo that keeps the fold")
    finally:
        _put_back(model, before)
    check_restored(model, before)


@pytest.fixture(scope="module")
def bf16_out(models, real_batch) -> Tensor:
    """The unfixed bf16 backbone's output (the reference the fixed runs compare to)."""
    return forward(models.get(BASE_BF16), real_batch.on("cuda", torch.bfloat16))


@pytest.fixture(scope="module")
def fp16_outs(models, real_batch, bf16_out) -> dict[str, Tensor]:
    """The fp16 backbone's output under each fix, and under a weight fold whose stream
    writers (embedders, text-refiner outputs) were left unscaled."""
    model = models.get(BASE_FP16)
    batch = real_batch.on("cuda", torch.float16)
    out = {}
    for mode, kw in FIXES.items():
        apply_fp16_fix(model, mode, **kw)
        out[mode] = forward(model, batch)
        clear_fp16_fix(model)
    apply_fp16_fix(model, "weights", c=C)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name in FOLD_MODULE.writer_scales(model, C):
            params[name].mul_(1.0 / C)
    out["writers_unscaled"] = forward(model, batch)
    models.free()
    return out


def test_fp16_fixes_match_the_fp32_residual_run(fp16_outs, bf16_out):
    ref = fp16_outs["fp32_residual"]
    for mode in ("online", "weights"):
        check_within(fp16_outs[mode], ref, bf16_out)
    must_fail(
        lambda: check_within(fp16_outs["writers_unscaled"], ref, bf16_out),
        "a weight fold that misses the stream writers",
    )


def test_fp16_fix_file(models, fix_file, real_batch, fp16_outs, bf16_out):
    if fix_file is None:
        pytest.skip("no fp16-fix file")
    record = read_fp16_fix(fix_file)
    assert record is not None, "the fp16-fix file carries no fold record"
    model = models.get(FILE_FP16)
    assert model.fp16_fix == "weights" and model.fold_scales[0] == record["c"]
    apply_fp16_fix(model, "none")
    assert model.fp16_fix == "weights" and model.fold_scales[0] == record["c"]
    out = forward(model, real_batch.on("cuda", torch.float16))
    models.free()
    check_within(out, fp16_outs["fp32_residual"], bf16_out)
