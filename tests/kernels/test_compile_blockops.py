"""Every block op traces under torch.compile(fullgraph=True), forward and backward,
and matches its eager call."""

import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry

BACKENDS = ("triton", "tilelang")


def _impl(op, backend, device, dtype):
    try:
        fn = registry.resolve(op, device, dtype, backend=backend)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))
    return getattr(fn, "__wrapped__", fn)


def _cases(device, dtype):
    seq = build_seqinfo(torch.tensor([[4, 6], [3, 5]]), torch.tensor([7, 11]))
    n = seq.n_joint
    seg = seq.seg_joint.to(device)
    d = 256
    x = torch.randn(n, d, device=device).to(dtype)
    w = torch.ones(d, device=device)
    mods = 0.1 * torch.randn(2, 6, d, device=device)
    mod = mods[:, 0]
    q = torch.randn(n, 2, 128, device=device).to(dtype)
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    pos = seq.pos_joint.to(device)
    w_ug = (torch.randn(512, d, device=device) / 16).to(dtype)
    w_o = (torch.randn(d, d, device=device) / 16).to(dtype)
    return {
        "rms_norm_mod": (x, w, mod, mods[:, 1], seg, 1e-6),
        "qk_norm_rope": (
            q,
            q.clone(),
            torch.ones(128, device=device),
            torch.ones(128, device=device),
            pos,
            bands,
            1e-6,
        ),
        "situ_glu": (torch.randn(n, 2 * d, device=device).to(dtype),),
        "gated_residual": (x, x.clone(), mod, seg, 1.0),
        "token_mse": (
            x[:, :128].contiguous(),
            x[:, 128:].contiguous(),
            seg,
            torch.ones(2, device=device),
            None,
        ),
        "linear_situ": (x, w_ug, None),
        "linear_gated_residual": (x, w_o, None, x.clone(), mod, seg, 1.0),
    }


def _sum(out):
    outs = out if isinstance(out, tuple) else (out,)
    return sum(o.float().sum() for o in outs)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "op",
    [
        "rms_norm_mod",
        "qk_norm_rope",
        "situ_glu",
        "gated_residual",
        "token_mse",
        "linear_situ",
        "linear_gated_residual",
    ],
)
def test_fullgraph(backend, op, device):
    torch._dynamo.reset()
    dtype = torch.float16
    fn = _impl(op, backend, device, dtype)
    args = _cases(device, dtype)[op]
    grads = [isinstance(a, torch.Tensor) and a.is_floating_point() for a in args]

    def leaves():
        return [
            a.detach().clone().requires_grad_(g) if isinstance(a, torch.Tensor) else a
            for a, g in zip(args, grads)
        ]

    eager_leaves = leaves()
    _sum(fn(*eager_leaves)).backward()
    compiled = torch.compile(lambda *a: _sum(fn(*a)), fullgraph=True, dynamic=False)
    comp_leaves = leaves()
    compiled(*comp_leaves).backward()
    for a, b in zip(eager_leaves, comp_leaves):
        if isinstance(a, torch.Tensor) and a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, rtol=2e-3, atol=2e-3)
