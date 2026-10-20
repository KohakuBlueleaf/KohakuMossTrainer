"""HostGemm, model-free: on bf16 / fp16 host tensors a linear, a batched matmul and a
convolution equal the 16-bit-input, fp32-accumulate GEMM, witnessed on integer inputs
where that GEMM is exact (the reference is the exact sum rounded once to the output
dtype); a variant that skips the 16-bit rounding of the activation fails; fp32 and
integer ops pass through untouched; `host_gemm` binds nothing where the host has
native 16-bit GEMMs."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss_kernels.hardware import host_gemm as hg
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


RANGES = {torch.bfloat16: (257, 1024), torch.float16: (2049, 4096)}
INNER = {torch.bfloat16: 128, torch.float16: 4}


def _activation(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    """fp32 integers most of which `dtype` cannot hold (bf16: 8 significant bits above
    256; fp16: 11 above 2048)."""
    g = torch.Generator().manual_seed(0)
    return torch.randint(*RANGES[dtype], shape, generator=g).float()


def _weight(*shape, top: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randint(-top, top + 1, shape, generator=g).float()


def assert_linear(gemm, dtype) -> None:
    k = INNER[dtype]
    x32 = _activation(64, k, dtype=dtype)
    w = _weight(96, k, top=8 if dtype == torch.bfloat16 else 2).to(dtype)
    x = x32.to(dtype)
    ref = (x.double() @ w.double().T).to(dtype)
    with hg.HostGemm():
        out = gemm(x32, x, w)
    assert out.dtype == dtype and torch.equal(out, ref)


def _host_linear(x32, x, w):
    return F.linear(x, w)


def _unrounded(x32, x, w):
    return F.linear(x32, w.float()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_is_the_16bit_input_fp32_gemm(dtype):
    assert_linear(_host_linear, dtype)
    must_fail(lambda: assert_linear(_unrounded, dtype), "activation not rounded")


@pytest.mark.parametrize("grad_mode", [torch.no_grad, torch.inference_mode])
def test_every_gemm_form_in_every_grad_mode(grad_mode):
    """Under inference mode the mode sees the composite ops (linear, conv2d, einsum,
    attention) before they decompose; under no_grad the decomposed ones."""
    a, b = _activation(3, 16, 64).to(torch.bfloat16), _weight(3, 64, 8).bfloat16()
    img, k = _activation(1, 4, 9, 9).bfloat16(), _weight(5, 4, 3, 3).bfloat16()
    x, w = _activation(16, 64).bfloat16(), _weight(24, 64).bfloat16()
    with grad_mode(), hg.HostGemm():
        out = torch.bmm(a, b)
        ein = torch.einsum("bij,bjk->bik", a, b)
        conv = F.conv2d(img, k, padding=1)
        lin = F.linear(x, w)
    ref = (a.double() @ b.double()).bfloat16()
    assert torch.equal(out, ref) and torch.equal(ein, ref)
    ref = F.conv2d(img.double(), k.double(), padding=1).bfloat16()
    assert conv.dtype == torch.bfloat16 and torch.equal(conv, ref)
    assert torch.equal(lin, (x.double() @ w.double().T).bfloat16())


def assert_every_form_upcast(grad_mode, monkeypatch) -> None:
    """Each of linear, bmm, einsum, conv2d and attention runs on fp32 inputs."""
    calls = []
    real_float = torch.Tensor.float

    def counted(self, *a, **kw):
        calls.append(self.dtype)
        return real_float(self, *a, **kw)

    q = torch.randn(1, 2, 8, 16).bfloat16()
    x, w = _activation(16, 64).bfloat16(), _weight(24, 64).bfloat16()
    img, k = _activation(1, 4, 9, 9).bfloat16(), _weight(5, 4, 3, 3).bfloat16()
    forms = (
        lambda: F.linear(x, w),
        lambda: torch.bmm(x[None], w.T[None]),
        lambda: torch.einsum("ij,kj->ik", x, w),
        lambda: F.conv2d(img, k, padding=1),
        lambda: F.scaled_dot_product_attention(q, q, q),
    )
    monkeypatch.setattr(torch.Tensor, "float", counted)
    for form in forms:
        calls.clear()
        with grad_mode(), hg.HostGemm():
            out = form()
        assert out.dtype == torch.bfloat16 and torch.bfloat16 in calls, form


@pytest.mark.parametrize("grad_mode", [torch.no_grad, torch.inference_mode])
def test_every_form_is_upcast(grad_mode, monkeypatch):
    assert_every_form_upcast(grad_mode, monkeypatch)


def test_decomposed_ops_alone_miss_inference_mode(monkeypatch):
    decomposed = {
        f for f in hg.GEMMS if "conv2d" not in str(f) and "linear" not in str(f)
    }
    monkeypatch.setattr(hg, "GEMMS", frozenset(decomposed))
    must_fail(
        lambda: assert_every_form_upcast(torch.inference_mode, monkeypatch),
        "inference mode reaches linear / conv2d before they decompose",
    )


def test_other_ops_pass_through():
    x, w = torch.randn(4, 8), torch.randn(3, 8)
    with hg.HostGemm():
        out = F.linear(x, w)
        ints = torch.ones(2, 2, dtype=torch.int64) @ torch.ones(2, 2, dtype=torch.int64)
        half_add = torch.ones(3, dtype=torch.bfloat16) + 1
    assert out.dtype == torch.float32 and torch.equal(out, F.linear(x, w))
    assert ints.dtype == torch.int64 and half_add.dtype == torch.bfloat16


def test_bound_only_without_native_gemms(monkeypatch):
    monkeypatch.setattr(hg, "native_half", lambda dtype: True)
    assert not isinstance(hg.host_gemm(torch.bfloat16), hg.HostGemm)
    monkeypatch.setattr(hg, "native_half", lambda dtype: dtype == torch.float32)
    assert isinstance(hg.host_gemm(torch.bfloat16), hg.HostGemm)
    assert not isinstance(hg.host_gemm(torch.float32), hg.HostGemm)
