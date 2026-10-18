"""fp8 / MXFP8 compute kernels (CUDA sm89+): activation quantizers bit-equal to the
eager reference; every GEMM path, and the registered op's forward / input / bias
gradients (fp8 and 16-bit dgrad), against the fp64 product with the quantized weight
(activations and gradients unquantized) next to PyTorch at the path's precision
(quantize, dequantize, matmul; F.linear for the 16-bit path); torch.compile(fullgraph);
the arch support check; the planner leaving out mx paths a partial K tile breaks."""

import pytest
import torch

from kohakumoss.quant.compute import quantize_pair
from kohakumoss_kernels import registry
from kohakumoss_kernels.errors import is_build_error
from kohakumoss_kernels.fp8 import (
    RECIPES,
    dequantize,
    fwd_reference,
    quantize,
    swizzle_mx,
)
from kohakumoss_kernels.fp8.op import dgrad_reference
from kohakumoss_kernels.hardware.arch import _cuda_arch, arch_of
from tests._util.error_stats import assert_no_regression
from tests._util.fp8 import deq64
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda


def _fp8_device() -> bool:
    return torch.cuda.is_available() and arch_of("cuda").has_fp8


needs_fp8 = pytest.mark.skipif(not _fp8_device(), reason="needs CUDA sm89+")
if _fp8_device():
    from kohakumoss_kernels.backends.triton.fp8 import linear as fp8_linear
    from kohakumoss_kernels.backends.triton.fp8.quantize import (
        quantize_act,
        quantize_mx_swizzled,
    )

DTYPES = (torch.bfloat16, torch.float16)
SHAPES = ((200, 384, 256), (1, 256, 512), (1000, 512, 384))
WHOLE_TILE_GEMMS = ("vendor", "fused")


def _x(m, k, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = torch.logspace(-3, 2, m, device="cuda")[:, None]
    x = torch.randn(m, k, device="cuda", generator=g) * scale
    if m > 2:
        x[1] = 0.0
    return x.to(dtype)


def _w(n, k, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(n, k, device="cuda", generator=g) * 0.05


@needs_fp8
@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_quantizer_bytes_equal_eager_reference(recipe, dtype):
    x = _x(300, 1024, dtype)
    x[2, :32] = torch.linspace(-1.75, 1.75, 32, device="cuda").to(dtype)
    q, s = quantize_act(x, recipe)
    rq, rs = quantize(x, recipe, "act")
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8)), recipe
    assert torch.equal(s, rs.to(s.device)), recipe
    q2, _ = quantize_act(x.t().contiguous().t(), recipe)
    assert torch.equal(q2.view(torch.uint8), rq.view(torch.uint8)), "strided input"


@needs_fp8
@pytest.mark.parametrize("dtype", DTYPES)
def test_swizzled_quantizer_equals_eager_reference(dtype):
    x = _x(300, 512, dtype)
    q, s = quantize_mx_swizzled(x)
    rq, rs = quantize(x, "mx", "act")
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s, swizzle_mx(rs))


def _one_per_path(recipe, m):
    seen, out = set(), []
    for cfg in fp8_linear.candidates(recipe, arch_of("cuda").sm, m):
        name = (cfg["gemm"], cfg.get("impl"))
        if name not in seen:
            seen.add(name)
            out.append(cfg)
    return out


def _path_cases():
    if not _fp8_device():
        return []
    return [
        (recipe, cfg, dtype)
        for recipe in RECIPES
        for cfg in _one_per_path(recipe, 256)
        for dtype in DTYPES
    ]


def _path_id(case):
    recipe, cfg, dtype = case
    return f"{recipe}-{cfg['gemm']}-{cfg.get('impl', '')}-{str(dtype)[6:]}"


PATHS = _path_cases()


@needs_fp8
@pytest.mark.budget(25)
@pytest.mark.parametrize("case", PATHS, ids=[_path_id(c) for c in PATHS])
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_path_forward_and_dgrad(case, m, n, k):
    recipe, cfg, dtype = case
    x = _x(m, k, dtype)
    w = _w(n, k)
    bias = (torch.randn(n, device="cuda") * 0.1).to(dtype)
    q, s, aux = quantize_pair(w, recipe)
    w64 = deq64(q, s, recipe)
    try:
        y = fp8_linear._fwd_runner(cfg, recipe)(x, q, s, aux, bias)
    except Exception as err:  # noqa: BLE001
        if not is_build_error(err):
            raise
        pytest.skip(f"{_path_id(case)} cannot run on this build (the probe drops it)")
    if cfg["gemm"] == "w16":
        eager = torch.nn.functional.linear(x, dequantize(q, s, recipe, dtype), bias)
    else:
        eager = fwd_reference(x, q, s, aux, bias, recipe)
    assert y.dtype == dtype and y.shape == (m, n)
    name = f"{_path_id(case)} {m}x{n}x{k}"
    ref = x.double() @ w64.T + bias.double()
    assert_no_regression(y, eager, ref, f"{name} fwd", test="fp8")

    dy = _x(m, n, dtype, seed=2)
    qt, st, auxt = quantize_pair(w.t(), recipe)
    dx = fp8_linear._dgrad_runner(cfg, recipe)(dy, q, s, qt, st, auxt)
    if cfg["gemm"] == "w16":
        eager = dgrad_reference(dy, q, s, None, None, None, recipe)
    else:
        eager = dgrad_reference(dy, q, s, qt, st, auxt, recipe)
    assert dx.shape == (m, k)
    assert_no_regression(dx, eager, dy.double() @ w64, f"{name} dgrad", test="fp8")


@needs_fp8
@pytest.mark.parametrize("recipe", RECIPES)
def test_error_check_rejects_misplaced_weight_scale(recipe):
    m, n, k = 256, 384, 512
    x = _x(m, k, torch.bfloat16)
    rows = torch.arange(n, device="cuda")[:, None] // (128 if recipe == "block" else 1)
    w = _w(n, k) * torch.where(rows % 2 == 0, 1.0, 256.0)
    q, s, aux = quantize_pair(w, recipe)
    ref = x.double() @ deq64(q, s, recipe).T
    eager = fwd_reference(x, q, s, aux, None, recipe)
    shifted = s.roll(1, 0) if s.numel() > 1 else s * 256
    cfg = fp8_linear.candidates(recipe, arch_of("cuda").sm, m)[0]
    aux2 = swizzle_mx(shifted) if recipe == "mx" else None
    broken = fp8_linear._fwd_runner(cfg, recipe)(x, q, shifted, aux2, None)

    def check():
        assert_no_regression(
            broken, eager, ref, f"{recipe} misplaced scale", test="fp8"
        )

    must_fail(check, f"{recipe}: weight scales on the wrong rows")


@needs_fp8
@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("dgrad", (True, False))
@pytest.mark.parametrize("dtype", DTYPES)
def test_registered_op_grads(recipe, dgrad, dtype, monkeypatch, tmp_path):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    op = registry.resolve("fp8_linear", "cuda", dtype, backend="triton", recipe=recipe)
    m, n, k = 333, 256, 384
    w = _w(n, k)
    q, s, aux = quantize_pair(w, recipe)
    qt, st, auxt = quantize_pair(w.t(), recipe) if dgrad else (None, None, None)
    x = _x(m, k, dtype).view(3, 111, k).requires_grad_(True)
    bias = (torch.randn(n, device="cuda") * 0.1).to(dtype).requires_grad_(True)
    y = op(x, q, s, aux, qt, st, auxt, bias, recipe)
    assert y.shape == (3, 111, n)
    g = _x(m, n, dtype, seed=4).view(3, 111, n)
    y.backward(g)
    x2 = x.detach().reshape(m, k)
    w64 = deq64(q, s, recipe)
    ref = x2.double() @ w64.T + bias.double()
    eager = fwd_reference(x2, q, s, aux, bias, recipe)
    assert_no_regression(y.reshape(m, n), eager, ref, f"op {recipe} fwd", test="fp8")
    g2 = g.reshape(m, n)
    eager = dgrad_reference(g2, q, s, qt, st, auxt, recipe)
    ref = g2.double() @ w64
    assert_no_regression(
        x.grad.reshape(m, k), eager, ref, f"op {recipe} dx", test="fp8"
    )
    assert_no_regression(
        bias.grad, g2.float().sum(0).to(dtype), g2.double().sum(0), "dbias"
    )


@needs_fp8
@pytest.mark.parametrize("recipe", RECIPES)
def test_fullgraph_compile(recipe):
    dtype = torch.bfloat16
    op = registry.resolve("fp8_linear", "cuda", dtype, backend="triton", recipe=recipe)
    w = _w(256, 384)
    q, s, aux = quantize_pair(w, recipe)
    qt, st, auxt = quantize_pair(w.t(), recipe)
    x = _x(80, 384, dtype).requires_grad_(True)

    def loss(t):
        return op(t, q, s, aux, qt, st, auxt, None, recipe).float().square().sum()

    fn = torch.compile(loss, fullgraph=True)
    out = fn(x)
    out.backward()
    xe = x.detach().clone().requires_grad_(True)
    out_e = loss(xe)
    out_e.backward()
    torch.testing.assert_close(out, out_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x.grad, xe.grad, rtol=1e-2, atol=1e-3)


@needs_fp8
@pytest.mark.parametrize("recipe", RECIPES)
def test_gradient_free_op_equals_autograd_op(recipe):
    dtype = torch.bfloat16
    w = _w(256, 384)
    q, s, aux = quantize_pair(w, recipe)
    x = _x(80, 384, dtype).requires_grad_(True)
    args = (q, s, aux, None, None, None, None, recipe)
    full = registry.resolve(
        "fp8_linear", "cuda", dtype, backend="triton", recipe=recipe
    )
    free = registry.resolve(
        "fp8_linear_infer", "cuda", dtype, backend="triton", recipe=recipe
    )
    y = full(x, *args)
    y_free = free(x, *args)
    assert y.grad_fn is not None and y_free.grad_fn is None
    assert torch.equal(y.detach(), y_free)
    s2 = s + 1 if recipe == "mx" else s * 2
    other = free(x, q, s2, swizzle_mx(s2) if recipe == "mx" else None, *args[3:])

    def same():
        assert torch.equal(other, y_free)

    must_fail(same, "a different weight scale")


@pytest.mark.parametrize("sm,ok", [(75, False), (86, False), (89, True), (120, True)])
def test_triton_arch_support(sm, ok):
    impl = registry._STATE.impls["fp8_linear"].get("triton")
    if impl is None:
        pytest.skip("triton backend not loaded")
    query = registry.OpQuery(
        "fp8_linear", _cuda_arch(sm // 10, sm % 10), torch.float16, (("recipe", "mx"),)
    )
    assert (impl.reason(query) is None) == ok


def _whole_tiles_only_on_whole_k(cfgs: list[dict], k: int) -> None:
    """No path reading whole 128-column mx tiles is offered for a partial K tile."""
    if k % 128:
        bad = [c for c in cfgs if c["gemm"] in WHOLE_TILE_GEMMS]
        assert not bad, f"k={k}: {bad[:2]}"


def test_mx_planner_leaves_out_whole_tile_paths_on_partial_k():
    linear = pytest.importorskip("kohakumoss_kernels.backends.triton.fp8.linear")
    for k in (96, 160, 256):
        _whole_tiles_only_on_whole_k(linear.candidates("mx", 120, 512, True, k), k)
    whole = linear.candidates("mx", 120, 512, True, 256)
    assert {c["gemm"] for c in whole} >= set(WHOLE_TILE_GEMMS)
    unfiltered = linear.candidates("mx", 120, 512, True)
    must_fail(
        lambda: _whole_tiles_only_on_whole_k(unfiltered, 96),
        "the planner offering vendor / fused mx paths at K = 96",
    )
