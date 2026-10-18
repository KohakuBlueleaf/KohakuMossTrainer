"""fp8 path availability: a GEMM path that cannot run on the device (e.g. row-wise
torch._scaled_mm on sm89 builds without it) is probed, dropped and logged once, and the
call runs on the next path; a live call never ends in the unsupported path's error."""

import logging

import pytest
import torch

from kohakumoss.quant.compute import quantize_pair
from kohakumoss_kernels import registry, status
from kohakumoss_kernels.errors import BackendUnavailable
from kohakumoss_kernels.fp8 import fwd_reference
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.tuning import tuner
from tests._util.error_stats import assert_no_regression
from tests._util.fp8 import deq64
from tests._util.logs import collecting

pytestmark = pytest.mark.cuda

needs_fp8 = pytest.mark.skipif(
    not (torch.cuda.is_available() and arch_of("cuda").has_fp8),
    reason="needs CUDA sm89+",
)
if torch.cuda.is_available() and arch_of("cuda").has_fp8:
    from kohakumoss_kernels.backends.triton.fp8 import linear as fp8_linear

UNSUPPORTED = "Rowwise scaling is not currently supported on your device"
VENDOR = {"gemm": "vendor"}


def _unsupported(*args, **kwargs):
    raise RuntimeError(UNSUPPORTED)


def _fresh(monkeypatch, tmp_path, mode: str) -> None:
    """Unsupported `_scaled_mm`, empty tables, and the vendor path first on every arch."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(torch, "_scaled_mm", _unsupported)
    planned = fp8_linear.candidates
    monkeypatch.setattr(
        fp8_linear,
        "candidates",
        lambda r, sm, m, fp8=True, k=None: [VENDOR, *planned(r, sm, m, fp8, k)],
    )
    monkeypatch.setattr(tuner, "_TABLES", {})
    fp8_linear._DEAD.clear()
    tuner.reset()
    tuner.set_mode(mode)


def _case():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(200, 384, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(256, 384, device="cuda", generator=g) * 0.05
    q, s, aux = quantize_pair(w, "row")
    return x, q, s, aux


@needs_fp8
@pytest.mark.parametrize("mode", ("off", "table"))
def test_unsupported_vendor_path_falls_through(mode, monkeypatch, tmp_path):
    before = tuner.get_mode()
    try:
        _fresh(monkeypatch, tmp_path, mode)
        x, q, s, aux = _case()
        y = fp8_linear.fwd(x, q, s, aux, None, "row")
        if mode == "off":
            dead = [d for d in fp8_linear._DEAD if dict(d[4])["gemm"] == "vendor"]
            assert dead, "the vendor path was not marked unavailable"
        ref = x.double() @ deq64(q, s, "row").T
        eager = fwd_reference(x, q, s, aux, None, "row")
        assert_no_regression(y, eager, ref, "fp8 row after vendor fallback", test="fp8")
    finally:
        tuner.set_mode(before)


def _vendor_only(recipe, sm, m, fp8=True, k=None):
    return [VENDOR]


@needs_fp8
def test_no_runnable_path_falls_back_to_the_eager_backend(monkeypatch, tmp_path):
    """Every path dead: the triton op raises BackendUnavailable and the registry binds
    the eager reference; a pinned triton op (no fallback) raises it."""
    before = tuner.get_mode()
    try:
        _fresh(monkeypatch, tmp_path, "off")
        monkeypatch.setattr(status, "_FAILED", {})
        monkeypatch.setattr(status, "_DISK", {})
        monkeypatch.setattr(status, "_LOADED", [True])
        monkeypatch.setattr(fp8_linear, "candidates", _vendor_only)
        x, q, s, aux = _case()
        args = (q, s, aux, None, None, None, None, "row")
        op = registry.resolve("fp8_linear", "cuda", torch.bfloat16, recipe="row")
        pinned = registry.resolve(
            "fp8_linear", "cuda", torch.bfloat16, backend="triton", recipe="row"
        )
        assert op.backend == "triton"
        with collecting(logging.WARNING) as logs:
            y = op(x, *args)
        warned = [r.getMessage() for r in logs.records if r.levelno >= logging.WARNING]
        assert any("fp8_linear" in m and "failed, using" in m for m in warned), warned
        assert op.backend == "eager"
        assert torch.equal(y, fwd_reference(x, q, s, aux, None, "row"))
        with pytest.raises(BackendUnavailable):
            pinned(x, *args)
    finally:
        tuner.set_mode(before)


@needs_fp8
def test_a_dead_tile_leaves_the_other_tiles(monkeypatch, tmp_path):
    """A Triton tile that cannot build is dropped alone: the path runs another tile."""
    before = tuner.get_mode()
    try:
        sm89 = fp8_linear.candidates("row", 89, 200)
        _fresh(monkeypatch, tmp_path, "off")
        real = fp8_linear.gemm_runner
        first = sm89[0]
        assert first["gemm"] == "triton", "sm89 row starts on the Triton GEMM"
        tiles = [c for c in sm89 if c["gemm"] == "triton"]

        def broken_first(cfg, recipe):
            if cfg == first:
                raise RuntimeError("out of resource: shared memory")
            return real(cfg, recipe)

        monkeypatch.setattr(fp8_linear, "gemm_runner", broken_first)
        monkeypatch.setattr(
            fp8_linear, "candidates", lambda r, sm, m, fp8=True, k=None: [*tiles, VENDOR]
        )  # fmt: skip
        x, q, s, aux = _case()
        fp8_linear.fwd(x, q, s, aux, None, "row")
        gemms = {dict(d[4])["gemm"] for d in fp8_linear._DEAD}
        assert gemms == {"triton"} and len(fp8_linear._DEAD) == 1, fp8_linear._DEAD
    finally:
        tuner.set_mode(before)


@needs_fp8
def test_without_the_probe_the_call_fails(monkeypatch, tmp_path):
    """Negative: the unprobed selection binds the vendor path and the live call raises."""
    before = tuner.get_mode()
    try:
        _fresh(monkeypatch, tmp_path, "off")
        monkeypatch.setattr(fp8_linear, "_probe", lambda *a, **k: True)
        x, q, s, aux = _case()
        with pytest.raises(RuntimeError, match=UNSUPPORTED):
            fp8_linear.fwd(x, q, s, aux, None, "row")
    finally:
        tuner.set_mode(before)
