"""Kernel-vs-compile binding: looked up in the call, measured only by retune_pending."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.compile import race
from kohakumoss_kernels.backends.compile.prewarm import prewarm_block_ops
from kohakumoss_kernels.backends.eager import ffn, norm, rope
from kohakumoss_kernels.tuning import tuner


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    mode = tuner.get_mode()
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    race.reset()
    yield
    tuner.set_mode(mode)
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    race.reset()


def _slow_kernel(x):
    for _ in range(50):
        y = ffn.situ_glu(x)
    return y


@pytest.mark.cuda
def test_call_never_times_and_retune_binds_winner(device, fresh):
    op = race.raced("test", "situ_glu", _slow_kernel, race.shape_key)
    x = torch.randn(2048, 8192, device=device)
    op(x)
    assert op.bound and next(iter(op.bound.values())) is op.impls["kernel"]
    assert op.pending, "first sight of a problem is queued, not timed"
    assert race.retune_pending() >= 1
    assert op.choice == "compile"
    torch.testing.assert_close(op(x), ffn.situ_glu(x))
    assert op.bound[race._quick_key((x,))] is op.impls["compile"]


@pytest.mark.cuda
@pytest.mark.parametrize("ctx", ["inference", "checkpoint"])
def test_retune_inside_untimed_context_stays_queued(device, fresh, ctx):
    """A retune reached under inference mode or inside a checkpointed forward times
    nothing and binds nothing; the next retune outside binds the measured winner."""
    op = race.raced("test", "situ_glu", _slow_kernel, race.shape_key)
    x = torch.randn(1024, 4096, device=device, requires_grad=True)
    op(x)
    assert op.pending

    def inner(t):
        assert race.retune_pending() == 0
        return t * 1

    if ctx == "inference":
        with torch.inference_mode():
            inner(torch.ones(1, device=device))
    else:
        checkpoint(inner, x, use_reentrant=False).sum().backward()
    assert op.pending and not tuner._CHOSEN
    assert race.retune_pending() == 1
    assert op.choice == "compile" and not op.pending


@pytest.mark.cuda
def test_tune_off_never_races(device, fresh):
    tuner.set_mode("off")
    op = race.raced("test", "situ_glu", _slow_kernel, race.shape_key)
    op(torch.randn(256, 512, device=device))
    assert not op.pending and race.retune_pending() == 0


def _registered(op, device, dtype):
    try:
        return registry.resolve(op, device, dtype, backend="triton")
    except registry.NoImplementationError:
        return registry.resolve(op, device, dtype, backend="tilelang")


def _block_inputs(device, dtype):
    seq = build_seqinfo(torch.tensor([[4, 6], [3, 5]]), torch.tensor([7, 11]))
    n, d = seq.n_joint, 256
    x = torch.randn(n, d, device=device).to(dtype)
    mod = 0.1 * torch.randn(2, d, device=device)
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    return x, mod, seq.seg_joint.to(device), seq.pos_joint.to(device), bands


def _block(ops, x, mod, seg, pos, bands):
    n = x.shape[0]
    y = ops["rms_norm_mod"](x, None, mod, mod, seg, 1e-6)
    q, k = ops["qk_norm_rope"](
        y.view(n, 2, 128), y.view(n, 2, 128).flip(-1).contiguous(),
        torch.ones(128, device=x.device), torch.ones(128, device=x.device), pos, bands, 1e-6,
    )  # fmt: skip
    a = ops["situ_glu"](torch.cat((q.reshape(n, -1), k.reshape(n, -1)), -1))
    return ops["gated_residual"](x, a, mod, seg, 1.0)


EAGER = {
    "rms_norm_mod": norm.rms_norm_mod,
    "qk_norm_rope": rope.qk_norm_rope,
    "situ_glu": ffn.situ_glu,
    "gated_residual": norm.gated_residual,
}


@pytest.mark.cuda
def test_checkpointed_block_on_fresh_cache(device, fresh):
    """Default kernels inside non-reentrant checkpointing, first step on a fresh cache."""
    dtype = torch.float16
    ops = {name: _registered(name, device, dtype) for name in EAGER}
    x, mod, seg, pos, bands = _block_inputs(device, dtype)
    xa = x.detach().clone().requires_grad_(True)
    out = checkpoint(_block, ops, xa, mod, seg, pos, bands, use_reentrant=False)
    out.float().square().sum().backward()
    xe = x.detach().clone().requires_grad_(True)
    _block(EAGER, xe, mod, seg, pos, bands).float().square().sum().backward()
    torch.testing.assert_close(xa.grad, xe.grad, rtol=2e-2, atol=2e-2)
    race.retune_pending()
    xb = x.detach().clone().requires_grad_(True)
    checkpoint(
        _block, ops, xb, mod, seg, pos, bands, use_reentrant=False
    ).float().square().sum().backward()
    torch.testing.assert_close(xb.grad, xe.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.cuda
def test_prewarm_binds_every_op(device, fresh):
    ops = {name: _registered(name, device, torch.float16) for name in EAGER}
    measured = prewarm_block_ops(
        ops, device, torch.float16, [[300, 200]], dim=256, heads=2, ffn_dim=512
    )
    assert measured == len(ops)
    for op in ops.values():
        assert op.bound and not op.pending


@pytest.mark.cuda
@pytest.mark.parametrize("ctx", [torch.no_grad, torch.inference_mode])
def test_no_grad_forward_on_fresh_cache(device, fresh, ctx):
    dtype = torch.float32 if device.type == "cuda" else torch.float32
    try:
        ops = {name: _registered(name, device, dtype) for name in EAGER}
    except registry.NoImplementationError as err:
        pytest.skip(str(err))
    x, mod, seg, pos, bands = _block_inputs(device, dtype)
    with ctx():
        out = _block(ops, x, mod, seg, pos, bands)
        ref = _block(EAGER, x, mod, seg, pos, bands)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
