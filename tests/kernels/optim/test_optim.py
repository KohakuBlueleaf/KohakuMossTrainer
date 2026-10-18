"""Optimizer step kernels vs the torch reference (kohakumoss_kernels.optim.reference)."""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.optim import looped_multi, new_8bit_state, reference
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail

BACKENDS = ("triton", "tilelang")
HP = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.01)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "pdtype,sdtype,mode",
    [
        (torch.float32, torch.float32, "nearest"),
        (torch.bfloat16, torch.bfloat16, "stochastic"),
        (torch.float16, torch.float16, "kahan"),
    ],
)
def test_adamw_multi(backend, pdtype, sdtype, mode, device):
    """One launch over a tensor list equals per-tensor reference steps (same seeds)."""
    fn = _impl("adamw_step_multi", backend, device, pdtype)
    sizes = (7, 1024, 3000, 33, 2048 * 3 + 5)
    ours = [_state(n, pdtype, sdtype, device, seed=i) for i, n in enumerate(sizes)]
    ref = [[t.clone() for t in s] for s in ours]
    hp = (1e-3, 0.9, 0.95, 1e-8, 0.01)
    refs64 = [_adamw64(*s[:4], hp, 2) for s in ours]
    ref_multi = looped_multi(reference.adamw_step)
    for step in (1, 2):
        cols = list(zip(*ours))
        rcols = list(zip(*ref))
        comps = list(cols[4]) if mode == "kahan" else None
        rcomps = list(rcols[4]) if mode == "kahan" else None
        fn(
            list(cols[0]),
            list(cols[1]),
            list(cols[2]),
            list(cols[3]),
            comps,
            *hp,
            step,
            mode,
            11,
        )
        ref_multi(
            list(rcols[0]),
            list(rcols[1]),
            list(rcols[2]),
            list(rcols[3]),
            rcomps,
            *hp,
            step,
            mode,
            11,
        )
    for a, b, r64 in zip(ours, ref, refs64):
        check_states(a, b, r64, mode)


def _impl(op, backend, device, dtype):
    """The backend's own kernel (not its kernel-vs-reference choice)."""
    try:
        fn = registry.resolve(op, device, dtype, backend=backend)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))
    return getattr(fn, "__wrapped__", fn)


def _state(n, pdtype, sdtype, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    p = torch.randn(n, generator=gen).to(device, pdtype)
    g = (0.1 * torch.randn(n, generator=gen)).to(device, pdtype)
    m = (0.01 * torch.randn(n, generator=gen)).to(device, sdtype)
    r = (0.01 * torch.rand(n, generator=gen)).to(device, sdtype)
    c = torch.zeros(n, device=device, dtype=pdtype)
    return p, g, m, r, c


def _adamw64(p, g, m, r, hp, steps):
    """The AdamW recursion in fp64 on fp64 copies (the precision reference)."""
    lr, b1, b2, eps, wd = hp
    p, g, m, r = (t.double() for t in (p, g, m, r))
    for s in range(1, steps + 1):
        m = m * b1 + g * (1 - b1)
        r = (r.square() * b2 + g.square() * (1 - b2)).sqrt()
        upd = (m / (1 - b1**s)) / (r / (1 - b2**s) ** 0.5 + eps)
        p = p - lr * (upd + wd * p)
    return p, m, r


def _value(state, mode):
    """The parameter value a step leaves: p, or p + comp in kahan mode."""
    return state[0].float() + state[4].float() if mode == "kahan" else state[0]


def check_states(ours, ref, ref64, mode):
    """p (with its kahan compensation), m and r of the kernel have no regression against the
    fp64 recursion, relative to the torch reference in the same dtypes."""
    got = (_value(ours, mode), ours[2], ours[3])
    eager = (_value(ref, mode), ref[2], ref[3])
    for name, a, e, r in zip("pmr", got, eager, ref64):
        assert_no_regression(a, e, r, name)


def check_adamw(fn, pdtype, sdtype, mode, device, steps=3):
    n = 5000
    ours = _state(n, pdtype, sdtype, device)
    ref = [t.clone() for t in ours]
    hp = (HP["lr"], HP["beta1"], HP["beta2"], HP["eps"], HP["wd"])
    ref64 = _adamw64(*ours[:4], hp, steps)
    for s in range(1, steps + 1):
        c_o = ours[4] if mode == "kahan" else None
        c_r = ref[4] if mode == "kahan" else None
        fn(*ours[:4], c_o, *hp, s, mode, s)
        reference.adamw_step(*ref[:4], c_r, *hp, s, mode, s)
    check_states(ours, ref, ref64, mode)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "pdtype,sdtype,mode",
    [
        (torch.float32, torch.float32, "nearest"),
        (torch.bfloat16, torch.float32, "nearest"),
        (torch.bfloat16, torch.bfloat16, "stochastic"),
        (torch.float16, torch.float16, "kahan"),
        (torch.bfloat16, torch.bfloat16, "kahan"),
    ],
)
def test_adamw(backend, pdtype, sdtype, mode, device):
    fn = _impl("adamw_step", backend, device, pdtype)
    check_adamw(fn, pdtype, sdtype, mode, device)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_negative_wrong_decay(backend, device):
    """Decay applied to the gradient (coupled L2) instead of the parameter is caught."""
    fn = _impl("adamw_step", backend, device, torch.float32)

    def coupled(p, g, m, r, c, lr, b1, b2, eps, wd, step, mode, seed):
        fn(p, g + wd * p, m, r, c, lr, b1, b2, eps, 0.0, step, mode, seed)

    must_fail(
        lambda: check_adamw(coupled, torch.float32, torch.float32, "nearest", device)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_stochastic_rounding_unbiased(backend, device):
    """Mean of many stochastic writebacks equals the fp32 value; nearest does not."""
    fn = _impl("adamw_step", backend, device, torch.bfloat16)

    def bias(mode):
        n = 4096
        base = torch.full((n,), 1.0, device=device, dtype=torch.bfloat16)
        acc = torch.zeros(n, device=device, dtype=torch.float64)
        trials = 64
        for t in range(trials):
            p = base.clone()
            g = torch.full((n,), 1.0, device=device, dtype=torch.bfloat16)
            m = torch.zeros(n, device=device, dtype=torch.float32)
            r = torch.zeros(n, device=device, dtype=torch.float32)
            fn(p, g, m, r, None, 1e-4, 0.0, 0.0, 1e-8, 0.0, 1, mode, 1000 + t)
            acc += p.double()
        return (acc / trials).mean().item() - (1.0 - 1e-4)

    assert abs(bias("stochastic")) < 2e-5
    must_fail(
        lambda: _assert(abs(bias("nearest")) < 2e-5), "nearest rounding is biased"
    )


def _assert(cond):
    assert cond


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_kahan_tracks_fp32(backend, device):
    """Kahan writeback of tiny updates follows an fp32 master; plain nearest stalls."""
    fn = _impl("adamw_step", backend, device, torch.float16)

    def drift(mode):
        n = 1024
        p = torch.ones(n, device=device, dtype=torch.float16)
        c = torch.zeros_like(p)
        master = torch.ones(n, device=device, dtype=torch.float64)
        m = torch.zeros(n, device=device, dtype=torch.float32)
        r = torch.zeros(n, device=device, dtype=torch.float32)
        g = torch.ones(n, device=device, dtype=torch.float16)
        for s in range(1, 201):
            fn(
                p,
                g,
                m,
                r,
                c if mode == "kahan" else None,
                1e-5,
                0.0,
                0.0,
                1e-8,
                0.0,
                s,
                mode,
                s,
            )
            master -= 1e-5
        return (
            ((p.double() + (c.double() if mode == "kahan" else 0)) - master)
            .abs()
            .max()
            .item()
        )

    assert drift("kahan") < 1e-5
    must_fail(lambda: _assert(drift("nearest") < 1e-5), "nearest stalls")


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("pdtype", [torch.float32, torch.bfloat16])
def test_adamw8bit(backend, pdtype, device):
    fn = _impl("adamw8bit_step", backend, device, pdtype)
    n = 3000
    p, g, _, _, _ = _state(n, pdtype, torch.float32, device)
    ours = (p.clone(), *new_8bit_state(p))
    ref = (p.clone(), *new_8bit_state(p))
    for s in range(1, 4):
        gs = g * (1 + 0.1 * s)
        fn(ours[0], gs, *ours[1:], None, 1e-3, 0.9, 0.95, 1e-8, 0.01, s, "nearest", 0)
        reference.adamw8bit_step(
            ref[0], gs, *ref[1:], None, 1e-3, 0.9, 0.95, 1e-8, 0.01, s, "nearest", 0
        )
    torch.testing.assert_close(ours[0].float(), ref[0].float(), rtol=1e-2, atol=1e-6)
    assert (ours[1].int() - ref[1].int()).abs().max() <= 1
    assert (ours[3].int() - ref[3].int()).abs().max() <= 1
    torch.testing.assert_close(ours[2], ref[2], rtol=1e-5, atol=0)
