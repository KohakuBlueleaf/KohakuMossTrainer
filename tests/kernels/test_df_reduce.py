"""`df` reduce mode: fp32 parameter gradients (and the loss) of cancelling sums are
correctly rounded against fp64.

Rows come in pairs whose products nearly cancel (no pair rounds alike), plus a small
residual block, so the exact sum is far below the summed magnitudes: an fp32 or Kahan
sum of the same terms,
a pair partial whose low part is dropped, and the fp32 eager composition all miss it.
The qk-norm case sits at position 0 (exact angles), so the reduction is what is judged;
`test_rope_angles` judges the angles at large positions.
"""

import sys

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import loss as eager_loss
from kohakumoss_kernels.backends.eager import norm as eager_norm
from kohakumoss_kernels.backends.eager import rope as eager_rope
from kohakumoss_kernels.precision import rounding_error, ulp_error
from kohakumoss_kernels.reduce import MODE
from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(MODE != "df", reason="KOHAKUMOSS_REDUCE is not df"),
]

BACKENDS = ("triton", "tilelang")
EXACT = 0.5 + 2.0**-6
ANGLE_RMS_ULP = 4.0


def _impl(op, backend, device, dtype):
    try:
        fn = registry.resolve(op, device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def _dtype(backend):
    return torch.float16 if backend == "tilelang" else torch.bfloat16


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _spread(shape, gen):
    """Normal values scaled by 2**U[-4, 4] (a wide magnitude spread)."""
    scale = torch.exp2(torch.randint(-4, 5, shape, generator=gen).float())
    return torch.randn(shape, generator=gen) * scale


def _paired(base, resid):
    return torch.cat((base, base, resid))


def _opposed(g, resid, gen):
    """g, then -g perturbed by 2^-4 relative noise (so no pair rounds alike), then the
    residual block."""
    noise = torch.randn(g.shape, generator=gen) * g.abs() / 16
    return torch.cat((g, -g + noise, resid))


def _grads(fn, args, need, gys, to64=False):
    """Leaf gradients of `need` args after backward of fn(*args) with `gys`."""
    leaves = []
    for a, n in zip(args, need):
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            a = a.detach().clone()
            a = a.double() if to64 else a
            a.requires_grad_(n)
        leaves.append(a)
    outs = fn(*leaves)
    outs = outs if isinstance(outs, tuple) else (outs,)
    torch.autograd.backward(outs, [g.to(o.dtype) for o, g in zip(outs, gys)])
    return [leaves[i].grad for i, n in enumerate(need) if n]


def _assert_rounded(outs, refs, limit, what):
    for i, (o, r) in enumerate(zip(outs, refs)):
        err = rounding_error(o, r)
        assert err <= limit, f"{what} grad {i}: {err:.3f} ULP > {limit}"


def _residual_case(device, dtype, lens=(300, 41, 700), d=256):
    gen = _gen(1)
    xs, os_, gs, segs = [], [], [], []
    for b, n in enumerate(lens):
        o = _spread((n, d), gen)
        g = _spread((n, d), gen)
        e_o = torch.randn(9, d, generator=gen) / 4
        e_g = torch.randn(9, d, generator=gen) / 4
        os_.append(_paired(o, e_o))
        gs.append(_opposed(g, e_g, gen))
        xs.append(torch.randn(2 * n + 9, d, generator=gen))
        segs.append(torch.full((2 * n + 9,), b, dtype=torch.int32))
    x = torch.cat(xs).to(device, dtype)
    o = torch.cat(os_).to(device, dtype)
    gy = torch.cat(gs).to(device, dtype)
    gate = torch.randn(len(lens), d, generator=gen).to(device)
    seg = torch.cat(segs).to(device)
    return (x, o, gate, seg, 1.0), (False, False, True, False, False), [gy]


def _norm_case(device, dtype, lens=(260, 33), d=384):
    gen = _gen(2)
    xs, gs, segs = [], [], []
    for b, n in enumerate(lens):
        x = torch.randn(n, d, generator=gen) * 3
        g = _spread((n, d), gen)
        xs.append(_paired(x, torch.randn(7, d, generator=gen)))
        gs.append(_opposed(g, torch.randn(7, d, generator=gen) / 4, gen))
        segs.append(torch.full((2 * n + 7,), b, dtype=torch.int32))
    x = torch.cat(xs).to(device, dtype)
    gy = torch.cat(gs).to(device, dtype)
    w = (1 + 0.3 * torch.randn(d, generator=gen)).to(device)
    shift = (0.1 * torch.randn(len(lens), d, generator=gen)).to(device)
    scale = (0.1 * torch.randn(len(lens), d, generator=gen)).to(device)
    seg = torch.cat(segs).to(device)
    args = (x, w, shift, scale, seg, 1e-6)
    return args, (False, True, True, True, False, False), [gy]


def _rope_case(device, dtype, n=200, heads=3, dh=128):
    gen = _gen(3)
    nb = dh // 8
    q, k = (
        _paired(
            torch.randn(n, heads, dh, generator=gen),
            torch.randn(5, heads, dh, generator=gen),
        )
        for _ in "qk"
    )
    pos = torch.zeros(2 * n + 5, 4)
    gq, gk = (
        _opposed(
            _spread((n, heads, dh), gen),
            torch.randn(5, heads, dh, generator=gen) / 4,
            gen,
        )
        for _ in "qk"
    )
    qw = (1 + 0.3 * torch.randn(dh, generator=gen)).to(device)
    kw = (1 + 0.3 * torch.randn(dh, generator=gen)).to(device)
    bands = (1 / (10000.0 ** (torch.arange(nb, dtype=torch.float32) / nb))).to(device)
    args = (
        q.to(device, dtype),
        k.to(device, dtype),
        qw,
        kw,
        pos.to(device),
        bands,
        1e-6,
    )
    gys = [gq.to(device, dtype), gk.to(device, dtype)]
    return args, (False, False, True, True, False, False, False), gys


def _loss_case(device, dtype, n=3000, c=16):
    gen = _gen(4)
    p = torch.randn(n, c, generator=gen) * 2
    t = torch.randn(n, c, generator=gen)
    e_p = torch.randn(11, c, generator=gen)
    e_t = torch.randn(11, c, generator=gen)
    seg = torch.cat(
        (
            torch.zeros(n, dtype=torch.int32),
            torch.ones(n, dtype=torch.int32),
            torch.full((11,), 2, dtype=torch.int32),
        )
    )
    weight = torch.tensor([1.0, -1.0, 0.25])
    p2 = p + torch.randn(n, c, generator=gen) / 16
    args = (
        torch.cat((p, p2, e_p)).to(device, dtype),
        _paired(t, e_t).to(device, dtype),
        seg.to(device),
        weight.to(device),
        None,
    )
    return args


CASES = {
    "gated_residual": (_residual_case, eager_norm.gated_residual, EXACT),
    "rms_norm_mod": (_norm_case, eager_norm.rms_norm_mod, EXACT),
    "qk_norm_rope": (_rope_case, eager_rope.qk_norm_rope, EXACT),
}


def _check_grads(fn, eager, args, need, gys, limit, what):
    ref = _grads(eager, args, need, gys, to64=True)
    _assert_rounded(_grads(fn, args, need, gys), ref, limit, what)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", sorted(CASES))
def test_parameter_grads_rounded(backend, op, device):
    case, eager, limit = CASES[op]
    dtype = _dtype(backend)
    fn = _impl(op, backend, device, dtype)
    args, need, gys = case(device, dtype)
    _check_grads(fn, eager, args, need, gys, limit, f"{backend} {op}")


@pytest.mark.parametrize("op", sorted(CASES))
def test_negative_fp32_eager(op, device):
    """The fp32 eager composition (plain fp32 sums, rounded phase) misses the bar."""
    case, eager, limit = CASES[op]
    args, need, gys = case(device, torch.float16)
    must_fail(lambda: _check_grads(eager, eager, args, need, gys, limit, "eager"), op)


def _module(backend, name):
    """A backend module the registry has loaded (no import of optional backends)."""
    return sys.modules[f"kohakumoss_kernels.backends.{backend}.{name}"]


def _lo_dropped(fn, axis):
    """`fn(part, ...)` with the low plane (index 1 on `axis`) of part zeroed first."""

    def broken(part, *rest, **kw):
        part.select(axis, 1).zero_()
        return fn(part, *rest, **kw)

    return broken


def _drop_lo(monkeypatch, backend):
    """Patch the backend's pair sums so the low planes of the partials are dropped."""
    blockops = _module(backend, "_blockops")
    match backend:
        case "triton":
            seg_reduce = blockops.SegPlan.reduce

            def reduce(self, part, *rest, **kw):
                part[:, :, 1].zero_()
                return seg_reduce(self, part, *rest, **kw)

            monkeypatch.setattr(blockops.SegPlan, "reduce", reduce)
            totals, axis = ("rope", "loss"), 2
        case _:
            sums = _lo_dropped(blockops.segment_sums, 3)
            for name in ("norm", "residual"):
                monkeypatch.setattr(_module(backend, name), "segment_sums", sums)
            totals, axis = ("rope", "loss"), 3
    for name in totals:
        monkeypatch.setattr(
            _module(backend, name), "total", _lo_dropped(blockops.total, axis)
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", sorted(CASES))
def test_negative_lo_dropped(backend, op, device, monkeypatch):
    """Partials kept as single fp32 values (low parts dropped) miss the bar."""
    case, eager, limit = CASES[op]
    dtype = _dtype(backend)
    fn = _impl(op, backend, device, dtype)
    args, need, gys = case(device, dtype)
    _drop_lo(monkeypatch, backend)
    must_fail(lambda: _check_grads(fn, eager, args, need, gys, limit, "lo"), op)


@pytest.mark.parametrize("backend", BACKENDS)
def test_loss_rounded(backend, device, monkeypatch):
    dtype = _dtype(backend)
    fn = _impl("token_mse", backend, device, dtype)
    args = _loss_case(device, dtype)
    ref = eager_loss.token_mse(*[a.double() if _fp(a) else a for a in args])

    def check(f):
        err = rounding_error(f(*args), ref)
        assert err <= EXACT, f"{backend} token_mse: {err:.3f} ULP"

    check(fn)
    must_fail(lambda: check(eager_loss.token_mse), "fp32 eager loss")
    _drop_lo(monkeypatch, backend)
    must_fail(lambda: check(fn), "loss lo dropped")


def _fp(a):
    return isinstance(a, torch.Tensor) and a.is_floating_point()


@pytest.mark.parametrize("backend", BACKENDS)
def test_rope_angles(backend, device):
    """At positions up to 4000 the q/k-norm weight gradients stay within a few fp32 ULP
    (RMS) of fp64: the phase rounding error is corrected. The spec's fp32 phase (off by
    up to 2^-13) is hundreds of ULP away."""
    dtype = _dtype(backend)
    fn = _impl("qk_norm_rope", backend, device, dtype)
    gen = _gen(6)
    n, heads, dh = 300, 3, 128
    nb = dh // 8
    q = torch.randn(n, heads, dh, generator=gen).to(device, dtype)
    k = torch.randn(n, heads, dh, generator=gen).to(device, dtype)
    pos = torch.randint(1000, 4000, (n, 4), generator=gen).float().to(device)
    bands = (1 / (10000.0 ** (torch.arange(nb, dtype=torch.float32) / nb))).to(device)
    w = (1 + 0.3 * torch.randn(dh, generator=gen)).to(device)
    args = (q, k, w, w.clone(), pos, bands, 1e-6)
    need = (False, False, True, True, False, False, False)
    gys = [torch.randn(n, heads, dh, generator=gen).to(device, dtype) for _ in "qk"]
    ref = _grads(eager_rope.qk_norm_rope, args, need, gys, to64=True)

    def check(f):
        for i, (o, r) in enumerate(zip(_grads(f, args, need, gys), ref)):
            err = ulp_error(o, r, torch.float32, "rms")
            assert err <= ANGLE_RMS_ULP, f"{backend} weight grad {i}: {err:.2f} ULP rms"

    check(fn)
    must_fail(lambda: check(eager_rope.qk_norm_rope), "fp32 phase")


def test_fp32_norm_rounded(device):
    """fp32 inputs (fp32 residual stream): the Triton norm's y and dx are rounded once
    from double-float; the fp32 eager composition is not."""
    fn = _impl("rms_norm_mod", "triton", device, torch.float32)
    gen = _gen(5)
    n, d = 300, 384
    x = (torch.randn(n, d, generator=gen) * 3).to(device)
    w = (1 + 0.3 * torch.randn(d, generator=gen)).to(device)
    shift = (0.1 * torch.randn(2, d, generator=gen)).to(device)
    scale = (0.1 * torch.randn(2, d, generator=gen)).to(device)
    seg = (torch.arange(n) >= 100).int().to(device)
    gy = torch.randn(n, d, generator=gen).to(device)
    args = (x, w, shift, scale, seg, 1e-6)
    need = (True, False, False, False, False, False)
    ref_y = eager_norm.rms_norm_mod(*[a.double() if _fp(a) else a for a in args])
    ref_dx = _grads(eager_norm.rms_norm_mod, args, need, [gy], to64=True)

    def check(f):
        err = rounding_error(f(*args).detach(), ref_y)
        assert err <= EXACT, f"y: {err:.3f} ULP"
        _assert_rounded(_grads(f, args, need, [gy]), ref_dx, EXACT, "fp32 norm")

    check(fn)
    must_fail(lambda: check(eager_norm.rms_norm_mod), "fp32 eager norm")
