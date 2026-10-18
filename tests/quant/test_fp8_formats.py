"""fp8 compute formats: the quantizer spec's bytes against an independent numpy formula
(e4m3 rounding from the format's bit layout), the scale layouts, and the vendor swizzle
against its index formula."""

import numpy as np
import pytest
import torch

from kohakumoss_kernels.fp8 import RECIPES, dequantize, quantize, swizzle_mx
from tests._util.fp8 import scale_numel
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _e4m3_table() -> tuple[np.ndarray, np.ndarray]:
    """Every finite non-negative e4m3fn value and its byte, ascending."""
    vals, codes = [], []
    for code in range(0x7F):
        e, m = code >> 3, code & 7
        v = 2.0**-6 * m / 8 if e == 0 else 2.0 ** (e - 7) * (1 + m / 8)
        vals.append(v)
        codes.append(code)
    order = np.argsort(vals, kind="stable")
    return np.array(vals)[order], np.array(codes)[order]


VALS, CODES = _e4m3_table()


def _e4m3_bytes(v: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even e4m3fn bytes of fp32 values, saturating at 448."""
    a = np.minimum(np.abs(v.astype(np.float64)), 448.0)
    hi = np.clip(np.searchsorted(VALS, a), 1, len(VALS) - 1)
    lo = hi - 1
    d_lo, d_hi = a - VALS[lo], VALS[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (CODES[hi] % 2 == 0))
    exact = VALS[np.minimum(hi, len(VALS) - 1)] == a
    idx = np.where(exact | pick_hi, hi, lo)
    code = CODES[idx]
    sign = np.signbit(v.astype(np.float32)).astype(np.int64) << 7
    return (code | sign).astype(np.uint8)


def _fp32_scale(amax: np.ndarray) -> np.ndarray:
    return (np.maximum(amax, np.float32(1e-12)) / np.float32(448.0)).astype(np.float32)


def _mx_exponent(amax: float) -> int:
    amax = max(float(amax), float(np.float32(1e-30)))
    e = -127
    while 2.0**e * 448.0 < amax:
        e += 1
    return e


def _reference(x: np.ndarray, recipe: str, role: str) -> tuple[np.ndarray, np.ndarray]:
    """(e4m3 bytes, scale) by the format definitions, block by block."""
    rows, k = x.shape
    q = np.zeros((rows, k), np.uint8)
    match recipe, role:
        case "tensor", _:
            s = _fp32_scale(np.abs(x).max(keepdims=True).reshape(1))
            return _e4m3_bytes((x / s[0]).astype(np.float32)), s
        case "row", _:
            s = _fp32_scale(np.abs(x).max(axis=1))
            return _e4m3_bytes((x / s[:, None]).astype(np.float32)), s
        case "block", "act":
            s = np.zeros((rows, k // 128), np.float32)
            for r in range(rows):
                for j in range(k // 128):
                    blk = x[r, j * 128 : (j + 1) * 128]
                    s[r, j] = _fp32_scale(np.abs(blk).max())
                    q[r, j * 128 : (j + 1) * 128] = _e4m3_bytes((blk / s[r, j]).astype(np.float32))  # fmt: skip
            return q, s
        case "block", "weight":
            nb = -(-rows // 128)
            s = np.zeros((nb, k // 128), np.float32)
            for i in range(nb):
                for j in range(k // 128):
                    blk = x[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
                    s[i, j] = _fp32_scale(np.abs(blk).max())
                    q[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = _e4m3_bytes((blk / s[i, j]).astype(np.float32))  # fmt: skip
            return q, s
        case "mx", _:
            s = np.zeros((rows, k // 32), np.uint8)
            for r in range(rows):
                for j in range(k // 32):
                    blk = x[r, j * 32 : (j + 1) * 32]
                    e = _mx_exponent(np.abs(blk).max())
                    s[r, j] = e + 127
                    q[r, j * 32 : (j + 1) * 32] = _e4m3_bytes((blk * 2.0**-e).astype(np.float32))  # fmt: skip
            return q, s
    raise ValueError(recipe)


def _operand(rows=7, k=256, seed=0) -> torch.Tensor:
    """Rows of very different magnitude, one all-zero row, and amax values on and next
    to the 1.75 * 2^E boundary of the mx exponent."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=g) * torch.logspace(-6, 3, rows)[:, None]
    x[1] = 0.0
    x[2, :32] = torch.linspace(-1.75, 1.75, 32)
    x[3, 32:64] = torch.linspace(-1.75, 1.75, 32) * (1 + 2.0**-7)
    x[4, 64:96] = torch.full((32,), 448.0 * 2.0**-130)
    return x.to(torch.bfloat16).float()


CASES = [(r, "act") for r in RECIPES] + [("block", "weight"), ("mx", "weight")]


@pytest.mark.parametrize("recipe,role", CASES)
def test_spec_bytes_match_independent_formula(recipe, role):
    x = _operand(rows=130 if role == "weight" else 7)
    q, s = quantize(x, recipe, role)
    rq, rs = _reference(x.numpy(), recipe, role)
    assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
    assert np.array_equal(q.view(torch.uint8).numpy(), rq), f"{recipe}/{role} values"
    assert np.array_equal(s.numpy(), rs), f"{recipe}/{role} scales"


def _round_to_nearest_mx(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g = x.float().view(x.shape[0], -1, 32)
    e = torch.round(torch.log2(g.abs().amax(-1).clamp_min(1e-30) / 448.0))
    q = (g / torch.exp2(e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view_as(x), (e.clamp_min(-127) + 127).to(torch.uint8)


def test_reference_rejects_broken_quantizers():
    x = _operand()
    rq, rs = _reference(x.numpy(), "mx", "act")
    q, s = _round_to_nearest_mx(x)

    def mx_check():
        assert np.array_equal(q.view(torch.uint8).numpy(), rq)
        assert np.array_equal(s.numpy(), rs)

    must_fail(mx_check, "mx exponent rounded to nearest instead of up")
    rq, rs = _reference(x.numpy(), "row", "act")
    s_col = (x.abs().amax(0).clamp_min(1e-12) / 448.0)[: x.shape[0]]
    q_col = (x / s_col[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)

    def row_check():
        assert np.array_equal(q_col.view(torch.uint8).numpy(), rq)

    must_fail(row_check, "row scales taken along the wrong axis")
    big = _operand(rows=64, k=1024, seed=5)
    rq, rs = _reference(big.numpy(), "row", "act")
    s = torch.from_numpy(rs)
    twice = (big / s[:, None]).clamp(-448, 448).half().to(torch.float8_e4m3fn)

    def rounded_once():
        assert np.array_equal(twice.view(torch.uint8).numpy(), rq)

    must_fail(rounded_once, "fp32 -> fp16 -> e4m3 (two roundings)")


@pytest.mark.parametrize("recipe,role", CASES)
def test_scale_layout(recipe, role):
    rows, k = (300, 512)
    x = _operand(rows=rows, k=k)
    _, s = quantize(x, recipe, role)
    expected = {
        ("tensor", "act"): ((1,), torch.float32),
        ("row", "act"): ((rows,), torch.float32),
        ("block", "act"): ((rows, k // 128), torch.float32),
        ("block", "weight"): ((-(-rows // 128), k // 128), torch.float32),
        ("mx", "act"): ((rows, k // 32), torch.uint8),
        ("mx", "weight"): ((rows, k // 32), torch.uint8),
    }[(recipe, role)]
    assert (tuple(s.shape), s.dtype) == expected
    assert s.numel() == scale_numel(recipe, rows, k, role)


@pytest.mark.parametrize("recipe,role", CASES)
def test_dequantize_is_q_times_scale(recipe, role):
    """dequantize against the per-element product of the reference bytes and scales."""
    x = _operand(rows=130, k=256)
    q, s = quantize(x, recipe, role)
    got = dequantize(q, s, recipe, torch.float32).double()
    vals = q.double()
    rows, k = q.shape
    r = torch.arange(rows)[:, None]
    c = torch.arange(k)[None, :]
    match recipe, role:
        case "tensor", _:
            per = s.double().expand(rows, k)
        case "row", _:
            per = s.double()[:, None].expand(rows, k)
        case "block", "act":
            per = s.double()[r, c // 128]
        case "block", _:
            per = s.double()[r // 128, c // 128]
        case _:
            per = torch.pow(2.0, s.double()[r, c // 32] - 127)
    want = (vals * per).float().double()
    assert torch.equal(got, want)


def _swizzle_index(r: int, c: int, col_tiles: int) -> int:
    return (r // 128) * col_tiles * 512 + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4  # fmt: skip


@pytest.mark.parametrize("rows,cols", [(128, 4), (200, 12), (1, 3), (300, 96)])
def test_swizzle_matches_index_formula(rows, cols):
    s = torch.randint(0, 255, (rows, cols), dtype=torch.uint8)
    flat = swizzle_mx(s)
    ct = -(-cols // 4)
    assert flat.numel() == -(-rows // 128) * 128 * ct * 4
    want = torch.zeros_like(flat)
    for r in range(rows):
        for c in range(cols):
            want[_swizzle_index(r, c, ct)] = s[r, c]
    assert torch.equal(flat, want)

    def broken():
        swapped = s.view(rows, cols)
        alt = torch.zeros_like(flat)
        for r in range(rows):
            for c in range(cols):
                i = (r // 128) * ct * 512 + (c // 4) * 512 + ((r % 128) // 32) * 128
                alt[i + (r % 32) * 4 + c % 4] = swapped[r, c]
        assert torch.equal(alt, want)

    if rows > 32:
        must_fail(broken, "row-major 32-row groups instead of the 32x4x4 interleave")
