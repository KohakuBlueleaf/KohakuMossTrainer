"""Fused LyCORIS kernels bound by the applier equal the eager bodies, forward and backward."""

import pytest
import torch
from _adcases import make_spec
from _adtoy import inputs, perturb, toy

from kohakumoss.adapters import apply

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


def _dtypes():
    if not torch.cuda.is_available():
        return [torch.float16]
    major, _ = torch.cuda.get_device_capability()
    return [torch.float16] if major < 8 else [torch.float16, torch.bfloat16]


def _run(name, mode, backend, dtype):
    model = toy().cuda().to(dtype)
    spec = make_spec(name, mode=mode, backend=backend)
    aset = apply(model, spec)
    perturb(aset)
    model.train()
    latent, text = (t.cuda().to(dtype) for t in inputs(n=64))
    y = model(latent, text)
    y.float().square().mean().backward()
    grads = {k: p.grad.float().clone() for k, p in aset.named_trainable()}
    return y.float().detach(), grads, aset


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", FUSED_CASES)
def test_fused_equals_eager(name, mode, dtype, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setenv("LYCORIS_KERNEL_BACKEND", "torch")
        y_e, g_e, _ = _run(name, mode, "torch", dtype)
    y_f, g_f, aset = _run(name, mode, "auto", dtype)
    assert aset.backend in ("triton", "tilelang")
    if name not in ("glora",) or mode == "merge":
        assert any(t.fused for t in aset.targets), "no layer took the fused path"
    tol = dict(rtol=3e-2, atol=3e-3)
    torch.testing.assert_close(y_f, y_e, **tol)
    for key in g_e:
        scale = g_e[key].abs().max().clamp_min(1e-6)
        torch.testing.assert_close(g_f[key] / scale, g_e[key] / scale, **tol, msg=key)
