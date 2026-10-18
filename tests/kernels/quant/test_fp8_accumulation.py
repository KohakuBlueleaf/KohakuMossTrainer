"""fp8 GEMM paths against plain PyTorch at the same quantized operands (CUDA sm89+):
every fp8 path the planner can time, at a deep K, is no worse in mean abs / rel error
(against fp64: the exact product of the quantized operands) than what PyTorch computes
(dequantize + F.linear, and PyTorch's fp8 GEMM `torch._scaled_mm` / `F.scaled_mm` with
the activation dtype as output where the device runs it). An accumulator rounded every
128 K fails it; on sm89 so do the Triton GEMM chaining the fp8 MMA accumulator and the
mx promote GEMM on the fp8 MMA."""

import pytest
import torch

from kohakumoss.quant.compute import quantize_pair
from kohakumoss_kernels.error_stats import stats
from kohakumoss_kernels.errors import is_build_error
from kohakumoss_kernels.fp8 import RECIPES, dequantize, fwd_reference, quantize
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.error_stats import record
from tests._util.fp8 import deq64
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
FP8 = torch.cuda.is_available() and arch_of("cuda").has_fp8
needs_fp8 = pytest.mark.skipif(not FP8, reason="needs CUDA sm89+")
if FP8:
    from kohakumoss_kernels.backends.triton.fp8 import linear as fp8_linear

M, N, K = 2048, 512, 8192
MEANS = ("abs_mean", "rel_mean")


def _operands(recipe, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M, K, device="cuda", generator=g).to(dtype)
    w = torch.randn(N, K, device="cuda", generator=g) * 0.02
    q, s, aux = quantize_pair(w, recipe)
    xq, xs = quantize(x, recipe, "act")
    return x, q, s, aux, deq64(xq, xs, recipe) @ deq64(q, s, recipe).T


def _torch_fp8(x, q, s, aux, recipe) -> torch.Tensor:
    """PyTorch's own fp8 GEMM of the quantized operands, output in x's dtype."""
    match recipe:
        case "tensor":
            xq, xs = quantize(x, recipe, "act")
            return torch._scaled_mm(
                xq, q.t(), scale_a=xs.reshape(()), scale_b=s.reshape(()),
                out_dtype=x.dtype,
            )  # fmt: skip
        case "row":
            xq, xs = quantize(x, recipe, "act")
            return torch._scaled_mm(
                xq, q.t(), scale_a=xs[:, None], scale_b=s[None, :], out_dtype=x.dtype
            )
        case "mx":
            return fp8_linear._vendor(recipe, x, q, s, aux, None)
    raise ValueError(f"PyTorch has no fp8 GEMM for recipe {recipe!r}")


def _rounded_every_128(x, q, s, recipe):
    """The fp8 product with the running sum rounded to x's dtype every 128 K."""
    xq, xs = quantize(x, recipe, "act")
    a = dequantize(xq, xs, recipe, torch.float32)
    b = dequantize(q, s, recipe, torch.float32)
    acc = torch.zeros(M, N, device="cuda", dtype=x.dtype)
    for k0 in range(0, K, 128):
        acc = (acc.float() + a[:, k0 : k0 + 128] @ b[:, k0 : k0 + 128].t()).to(x.dtype)
    return acc


def _pytorch(x, q, s, aux, recipe) -> list[torch.Tensor]:
    """Plain PyTorch results: dequantize + F.linear, and PyTorch's fp8 GEMM where the
    recipe has one and the device runs it."""
    outs = [fwd_reference(x, q, s, aux, None, recipe)]
    if recipe in ("tensor", "row", "mx"):
        try:
            outs.append(_torch_fp8(x, q, s, aux, recipe))
        except Exception as err:  # noqa: BLE001
            if not is_build_error(err):
                raise
    return outs


def _assert_no_worse(y, refs, ref64, name):
    """`y`'s mean abs and rel errors are at most the largest PyTorch result's."""
    got = stats(y, ref64)
    bounds = [stats(r, ref64) for r in refs]
    loosest = {s: max(b[s] for b in bounds) for s in bounds[0] if s != "finite"}
    record("fp8_accumulation", name, {"kernel": got, "eager": loosest})
    worse = [s for s in MEANS if got[s] > loosest[s]]
    assert not worse, f"{name}: worse than PyTorch on {worse}: {got} vs {loosest}"


def _paths(recipe):
    seen, out = set(), []
    for cfg in fp8_linear.candidates(recipe, arch_of("cuda").sm, M, True, K):
        name = cfg["gemm"]
        if name == "w16" or name in seen:
            continue
        seen.add(name)
        out.append(cfg)
    return out


CASES = [(r, d) for r in RECIPES for d in (torch.float16, torch.bfloat16)]


@needs_fp8
@pytest.mark.parametrize("recipe,dtype", CASES)
def test_paths_no_worse_than_pytorch_at_the_same_operands(recipe, dtype):
    x, q, s, aux, ref64 = _operands(recipe, dtype)
    refs = _pytorch(x, q, s, aux, recipe)
    for cfg in _paths(recipe):
        try:
            y = fp8_linear._fwd_runner(cfg, recipe)(x, q, s, aux, None)
        except Exception as err:  # noqa: BLE001
            if not is_build_error(err):
                raise
            continue
        _assert_no_worse(y, refs, ref64, f"{recipe}/{str(dtype)[6:]}/{cfg['gemm']}")
    broken = _rounded_every_128(x, q, s, recipe)
    must_fail(
        lambda: _assert_no_worse(broken, refs, ref64, "rounded every 128"),
        "an accumulator rounded to 16 bits every 128 K",
    )


@needs_fp8
@pytest.mark.skipif(not FP8 or arch_of("cuda").sm >= 90, reason="sm89 only")
@pytest.mark.parametrize("dtype", (torch.float16,))
def test_sm89_chained_fp8_accumulator_fails(dtype):
    """fp16 operands: PyTorch's fp16 GEMM is finer than Ada's chained fp8 MMA sum."""
    x, q, s, aux, ref64 = _operands("row", dtype)
    refs = _pytorch(x, q, s, aux, "row")
    tile = {"gemm": "triton", "bm": 128, "bn": 128, "bk": 128, "warps": 8, "stages": 3}
    promoted = fp8_linear._fwd_runner({**tile, "acc": "promote"}, "row")
    _assert_no_worse(promoted(x, q, s, aux, None), refs, ref64, "sm89 promoted")
    chained = fp8_linear._fwd_runner({**tile, "acc": "mma"}, "row")(x, q, s, aux, None)
    must_fail(
        lambda: _assert_no_worse(chained, refs, ref64, "sm89 chained"),
        "the fp8 MMA accumulator chained across K on sm89",
    )


@needs_fp8
@pytest.mark.skipif(not FP8 or arch_of("cuda").sm >= 90, reason="sm89 only")
def test_sm89_mx_fp8_mma_fails():
    """fp16 operands: Ada's fp8 MMA sums each 32-deep block coarser than fp32; the mx
    promote path runs the fp16 MMA on the exactly converted operands."""
    x, q, s, aux, ref64 = _operands("mx", torch.float16)
    refs = _pytorch(x, q, s, aux, "mx")
    tile = {"gemm": "promote", "bm": 128, "bn": 128, "bk": 0, "warps": 8, "stages": 4}
    upcast = fp8_linear._fwd_runner({**tile, "acc": "upcast"}, "mx")
    _assert_no_worse(upcast(x, q, s, aux, None), refs, ref64, "sm89 mx upcast")
    fp8_mma = fp8_linear._fwd_runner(tile, "mx")(x, q, s, aux, None)
    must_fail(
        lambda: _assert_no_worse(fp8_mma, refs, ref64, "sm89 mx fp8 MMA"),
        "the mx promote path on the fp8 MMA on sm89",
    )


def _all_promote(cfgs):
    for cfg in cfgs:
        if cfg["gemm"] in ("triton", "fused"):
            assert cfg.get("acc") == "promote", cfg


def _promote_upcast(cfgs):
    for cfg in cfgs:
        if cfg["gemm"] == "promote":
            assert cfg.get("acc") == "upcast", cfg


@needs_fp8
def test_planner_tiles_promote():
    """Every Triton / fused tile the planner offers promotes into fp32; on sm89 every mx
    promote tile runs the fp16 MMA."""
    for recipe in RECIPES:
        for sm in (89, 120):
            _all_promote(fp8_linear.candidates(recipe, sm, 4096))
    chained = [{**c, "acc": "mma"} for c in fp8_linear.candidates("row", 89, 4096)]
    must_fail(lambda: _all_promote(chained), "tiles chaining the MMA accumulator")
    sm89_mx = fp8_linear.candidates("mx", 89, 4096)
    _promote_upcast(sm89_mx)
    fp8_mma = [{k: v for k, v in c.items() if k != "acc"} for c in sm89_mx]
    must_fail(lambda: _promote_upcast(fp8_mma), "sm89 mx promote tiles on the fp8 MMA")
