"""Fused linear + epilogue ops vs the composition of eager primitives (error stats
against fp64)."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import ffn, norm, rope
from tests._util.negative import must_fail
from tests.kernels.blocks._blockops_harness import check

try:
    from kohakumoss_kernels.backends.triton.blocks import fused_linear as triton_fl
except ImportError:
    triton_fl = None
try:
    from kohakumoss_kernels.backends.tilelang.blocks import fused_linear as tilelang_fl
except ImportError:
    tilelang_fl = None

PATHS = ("fused", "split")
MODULES = {"triton": triton_fl, "tilelang": tilelang_fl}


def check_both(fn, eager, args, grads, dtype, name):
    """The training path (forward + backward) and the no-grad path (inference, the
    pre-activations kept in fp32 instead of saved) both have no regression."""
    check(fn, eager, args, grads, dtype, name)
    with torch.no_grad():
        check(fn, eager, args, grads, dtype, f"{name}-nograd", backward=False)


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


@pytest.mark.gpu
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
        args = (y, w, b)
        check_both(fn, eager_situ, args, (True, True, True), dtype, f"{backend}-{path}")
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


@pytest.mark.gpu
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
        gate = (0.5 * torch.randn(len(lens), 6, d)).to(device, torch.float32)[:, 2]
        args = (a, w, None, x, gate, seg.to(device), r)
        grads = (True, True, False, True, True, False, False)
        check_both(fn, eager_res, args, grads, dtype, f"{backend}-{path}")
    finally:
        mod.force_path(None)


@pytest.mark.gpu
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
        check_both(fn, eager_qk, args, grads, dtype, f"{backend}-{path}")
    finally:
        mod.force_path(None)


def _master_case(op, device, dtype):
    """(eager reference, args, grads) with 16-bit activations and fp32 master weights."""
    g = torch.Generator().manual_seed(7)
    m, k, d = 300, 256, 384
    y = torch.randn(m, k, generator=g).to(device, dtype)
    w = (torch.randn(2 * d, k, generator=g) / k**0.5).to(device)
    match op:
        case "linear_situ":
            b = (0.1 * torch.randn(2 * d, generator=g)).to(device)
            return eager_situ, (y, w, b), (True, True, True)
        case "linear_gated_residual":
            seg = (torch.arange(m) >= 120).int().to(device)
            x = torch.randn(m, d, generator=g).to(device, dtype)
            gate = torch.randn(2, d, generator=g).to(device)
            args = (y, w[:d], None, x, gate, seg, 1.0)
            return eager_res, args, (True, True, False, True, True, False, False)
    seq = build_seqinfo(torch.tensor([[4, 6], [3, 3]]), torch.tensor([20, 133]))
    y = torch.randn(seq.n_joint, k, generator=g).to(device, dtype)
    qw = (1 + 0.3 * torch.randn(128, generator=g)).to(device)
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    args = (y, w[:d], w[d:], qw, qw.clone(), seq.pos_joint.to(device), bands, 1e-6, 3)
    return eager_qk, args, (True, True, True, True, True, False, False, False, False)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "op", ("linear_situ", "linear_gated_residual", "linear_qk_rope")
)
def test_fp32_master_weights(backend, path, op, device):
    """Full finetune: fp16 activations with fp32 master weights. The matmuls take the
    weights cast to fp16; the weight gradients come back fp32 with no regression."""
    dtype = torch.float16
    fn = _impl(op, device, dtype, backend)
    mod = _force(backend, path)
    try:
        eager, args, grads = _master_case(op, device, dtype)
        check(fn, eager, args, grads, dtype, f"{backend}-{path}-{op}-master")
    finally:
        mod.force_path(None)


def _autocast(fn, device):
    """fn run under fp16 autocast (the trainer's mixed-precision forward)."""

    def run(*args):
        with torch.autocast(device.type, dtype=torch.float16):
            return fn(*args)

    return run


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "op", ("linear_situ", "linear_gated_residual", "linear_qk_rope")
)
def test_fp32_activations_under_autocast(backend, path, op, device):
    """fp32 activations and fp32 masters under fp16 autocast: the matmuls run on fp16
    operands (as autocast does for a linear), the backward outside autocast sees one
    dtype per matmul, and the gradients come back fp32."""
    fn = _impl(op, device, torch.float16, backend)
    mod = _force(backend, path)
    try:
        eager, args, grads = _master_case(op, device, torch.float32)
        check(
            _autocast(fn, device),
            _autocast(eager, device),
            args,
            grads,
            torch.float16,
            f"{backend}-{path}-{op}-autocast",
        )
    finally:
        mod.force_path(None)


@pytest.mark.gpu
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
