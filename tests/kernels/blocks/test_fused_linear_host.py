"""The Triton and TileLang fused linear ops run one host code (`backends/_fused_linear.py`):
the same public functions, op bodies and autograd for both, and the qk norm weights
enter the op as fp32 on both (the TileLang side of the former drift)."""

from dataclasses import replace

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends import _fused_linear as shared
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
PUBLIC = ("linear_situ", "linear_gated_residual", "linear_qk_rope")
PARTS = ("situ_fwd", "situ_bwd", "gres_fwd", "gres_bwd", "qk_fwd", "qk_bwd")


def _modules() -> list:
    registry.load_backends()
    mods = []
    for backend in ("triton", "tilelang"):
        mod = pytest.importorskip(
            f"kohakumoss_kernels.backends.{backend}.blocks.fused_linear"
        )
        mods.append(mod)
    return mods


def _code(fn):
    return getattr(fn, "__func__", fn).__code__


def check_shared(mods) -> None:
    """Every backend's public functions and op bodies are the shared host code."""
    for mod in mods:
        for name in PUBLIC:
            assert _code(getattr(mod, name)) is _code(
                getattr(shared.FusedLinear, name)
            ), f"{mod.__name__}.{name} is not the shared host code"
    bodies = [{p: m.FUSED.bodies[p].__code__ for p in PARTS} for m in mods]
    assert all(b == bodies[0] for b in bodies), "op bodies differ between backends"


def test_both_backends_share_the_host_code():
    check_shared(_modules())


def _diverged_qk(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads):
    """A backend-local copy of the qk entry (the former Triton side)."""
    return q_weight.contiguous(), k_weight.contiguous()


class _Diverged:
    """A backend module keeping its own copy of `linear_qk_rope`."""

    def __init__(self, mod):
        self.__name__ = f"{mod.__name__} (diverged)"
        self.FUSED = mod.FUSED
        self.linear_situ = mod.linear_situ
        self.linear_gated_residual = mod.linear_gated_residual
        self.linear_qk_rope = _diverged_qk


def test_negative_diverged_copy():
    mods = _modules()
    must_fail(lambda: check_shared([mods[0], _Diverged(mods[1])]), "diverged copy")


def check_norm_weights_fp32(entry, fl) -> None:
    """The qk op receives the norm weights in fp32 whatever their storage dtype, on the
    eager direct path and through the custom op."""
    seen = []

    def recorder(y, wq, wk, qw, kw, pos, bands, eps, heads, save):
        seen.append((qw.dtype, kw.dtype))
        m, d = y.shape[0], wq.shape[0]
        e = y.new_empty(0)
        return y.new_zeros(m, d), y.new_zeros(m, d), e, e, e, e

    real, fl._qk_fwd = fl._qk_fwd, recorder
    eager = fl.eager["qk"]
    fl.eager["qk"] = replace(eager, fwd=recorder)
    try:
        y = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        nw = torch.ones(4, dtype=torch.bfloat16)
        for dispatch in (False, True):
            fl.set_dispatch(dispatch)
            entry(y, w, w, nw, nw, torch.zeros(4, 1), torch.ones(2), 1e-6, 2)
    finally:
        fl._qk_fwd, fl.eager["qk"] = real, eager
        fl.set_dispatch(False)
    assert seen == [(torch.float32, torch.float32)] * 2, seen


def test_qk_norm_weights_enter_as_fp32():
    for mod in _modules():
        check_norm_weights_fp32(mod.linear_qk_rope, mod.FUSED)


def test_negative_storage_dtype_norm_weights():
    fl = _modules()[0].FUSED

    def old_entry(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads):
        args = (q_weight.contiguous(), k_weight.contiguous(), pos, bands, eps)
        return fl._forward("qk", y, wq, wk, *args, heads, False)

    must_fail(lambda: check_norm_weights_fp32(old_entry, fl), "storage dtype passed")
