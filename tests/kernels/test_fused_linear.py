"""Fused linear + epilogue ops vs the composition of eager primitives (fp64 bar)."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import ffn, norm, rope
from tests._util.negative import must_fail
from tests.kernels._blockops_harness import check

try:
    from kohakumoss_kernels.backends.triton import fused_linear as triton_fl
except ImportError:
    triton_fl = None
try:
    from kohakumoss_kernels.backends.tilelang import fused_linear as tilelang_fl
except ImportError:
    tilelang_fl = None

PATHS = ("fused", "split")
QK_FACTOR = 1.3
MODULES = {"triton": triton_fl, "tilelang": tilelang_fl}


def _impl(op, device, dtype, backend):
    try:
        fn = registry.resolve(op, device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def _force(backend, path):
    mod = MODULES[backend]
    mod.force_path({"path": "split"} if path == "split" else mod.FUSED_TEST_CFG)
    return mod


def eager_situ(y, w, b):
    return ffn.situ_glu(
        F.linear(y, w.to(y.dtype), None if b is None else b.to(y.dtype))
    )


def eager_res(a, w, b, x, gate, seg, r):
    o = F.linear(a, w.to(a.dtype), None if b is None else b.to(a.dtype))
    return norm.gated_residual(x, o, gate, seg, r)


def eager_qk(y, wq, wk, qw, kw, pos, bands, eps, heads):
    q = F.linear(y, wq.to(y.dtype)).unflatten(-1, (heads, -1))
    k = F.linear(y, wk.to(y.dtype)).unflatten(-1, (heads, -1))
    return rope.qk_norm_rope(q, k, qw, kw, pos, bands, eps)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,k,f", [(300, 256, 384), (1031, 1024, 2048)])
def test_linear_situ(backend, path, dtype, m, k, f, device):
    fn = _impl("linear_situ", device, dtype, backend)
    mod = _force(backend, path)
    try:
        y = torch.randn(m, k).to(device, dtype)
        w = (torch.randn(2 * f, k) / k**0.5 * 3).to(device, dtype)
        b = (0.1 * torch.randn(2 * f)).to(device, dtype)
        check(fn, eager_situ, (y, w, b), (True, True, True), dtype, f"{backend}-{path}")
        check(
            fn,
            eager_situ,
            (y, w, None),
            (True, False, False),
            dtype,
            f"{backend}-{path}-nb",
        )
    finally:
        mod.force_path(None)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("r", [1.0, 0.25])
def test_linear_gated_residual(backend, path, dtype, r, device):
    fn = _impl("linear_gated_residual", device, dtype, backend)
    mod = _force(backend, path)
    try:
        lens = (100, 3, 200)
        n, k, d = sum(lens), 512, 384
        seg = torch.cat(
            [torch.full((c,), i, dtype=torch.int32) for i, c in enumerate(lens)]
        )
        a = torch.randn(n, k).to(device, dtype)
        w = (torch.randn(d, k) / k**0.5).to(device, dtype)
        x = (torch.randn(n, d) * 4).to(device, dtype)
        gate = (0.5 * torch.randn(len(lens), d)).to(device, torch.float32)
        args = (a, w, None, x, gate, seg.to(device), r)
        grads = (True, True, False, True, True, False, False)
        check(fn, eager_res, args, grads, dtype, f"{backend}-{path}", factor=QK_FACTOR)
    finally:
        mod.force_path(None)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_qk_rope(backend, path, dtype, device):
    fn = _impl("linear_qk_rope", device, dtype, backend)
    mod = _force(backend, path)
    try:
        heads, dh, k = 3, 128, 256
        seq = build_seqinfo(torch.tensor([[4, 6], [3, 3]]), torch.tensor([20, 133]))
        y = torch.randn(seq.n_joint, k).to(device, dtype)
        wq = (torch.randn(heads * dh, k) / k**0.5).to(device, dtype)
        wk = (torch.randn(heads * dh, k) / k**0.5).to(device, dtype)
        qw = (1 + 0.3 * torch.randn(dh)).to(device, torch.float32)
        kw = (1 + 0.3 * torch.randn(dh)).to(device, torch.float32)
        bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
        args = (y, wq, wk, qw, kw, seq.pos_joint.to(device), bands, 1e-6, heads)
        grads = (True, True, True, True, True, False, False, False, False)
        check(fn, eager_qk, args, grads, dtype, f"{backend}-{path}", factor=QK_FACTOR)
    finally:
        mod.force_path(None)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_negative_gate_dropped(backend, device):
    """A fused residual that forgets the gate must be caught."""
    dtype = torch.float16
    fn = _impl("linear_gated_residual", device, dtype, backend)
    mod = _force(backend, "fused")
    try:
        n, k, d = 64, 256, 128
        seg = torch.zeros(n, dtype=torch.int32, device=device)
        a = torch.randn(n, k).to(device, dtype)
        w = (torch.randn(d, k) / k**0.5).to(device, dtype)
        x = torch.randn(n, d).to(device, dtype)
        gate = (2 + torch.randn(1, d)).to(device, torch.float32)

        def broken(a, w, b, x, gate, seg, r):
            return fn(a, w, b, x, None, None, r)

        args = (a, w, None, x, gate, seg, 1.0)
        grads = (True, False, False, True, False, False, False)
        must_fail(lambda: check(broken, eager_res, args, grads, dtype, "broken"))
    finally:
        mod.force_path(None)
