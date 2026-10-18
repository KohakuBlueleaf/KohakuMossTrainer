"""Card probe of the bench matrix: the measured dense GEMM rate, the tensor-core MMA peak
(the compute ceilings a run is held against) and the host link's copy bandwidth (the
transfer ceiling), written as JSON.

    python -m kohakumoss_bench.measure.probe --out probe.json   (on the visible card 0)

GEMM: n x n x n matmuls (fp32 accumulation, PyTorch's default) in bf16 where the card
has it and fp16, and fp8 e4m3 `torch._scaled_mm` (tensor-wise scales) on sm89+. MMA
peak: a Triton loop of register-resident `tl.dot` tiles with an fp32 accumulator (no
memory traffic), same dtypes. Link: 1 GiB copies host -> device and device -> host,
pinned and pageable. Device time from events, best of the repeats.
See docs/performance/bench-matrix.md#host-runner.
"""

import argparse
import json
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

MMA_TILE = (64, 128, 32)
MMA_ITERS = 4096
MMA_WAVES = 4
MMA_WARPS = 4

if triton is not None:

    @triton.jit
    def _mma_loop(
        a_ptr,
        b_ptr,
        out_ptr,
        ITERS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        m = tl.arange(0, BM)
        n = tl.arange(0, BN)
        k = tl.arange(0, BK)
        a = tl.load(a_ptr + m[:, None] * BK + k[None, :])
        b = tl.load(b_ptr + k[:, None] * BN + n[None, :])
        b2 = tl.load(b_ptr + BK * BN + k[:, None] * BN + n[None, :])
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(ITERS):
            acc = tl.dot(a, b, acc)
            acc2 = tl.dot(a, b2, acc2)
        out = out_ptr + tl.program_id(0) * BM * BN + m[:, None] * BN + n[None, :]
        tl.store(out, acc + acc2)


GEMM_N = 8192
REPEATS = 10
LINK_BYTES = 1 << 30


def _accel(device: torch.device):
    return torch.get_device_module(device)


def _timed_ms(fn, device: torch.device, repeats: int) -> float:
    """Best device milliseconds of `fn` over `repeats` calls (after one warm call)."""
    mod = _accel(device)
    fn()
    mod.synchronize(device)
    best = float("inf")
    for _ in range(repeats):
        start, end = mod.Event(enable_timing=True), mod.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def gemm_tflops(device: torch.device, dtype: torch.dtype, n: int = GEMM_N) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    ms = _timed_ms(lambda: a @ b, device, REPEATS)
    return 2 * n**3 / (ms * 1e-3) / 1e12


def fp8_tflops(device: torch.device, n: int = GEMM_N) -> float:
    a = torch.randn(n, n, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(n, n, device=device).to(torch.float8_e4m3fn).t()
    one = torch.ones((), device=device)

    def run():
        return torch._scaled_mm(
            a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16
        )

    ms = _timed_ms(run, device, REPEATS)
    return 2 * n**3 / (ms * 1e-3) / 1e12


def mma_tflops(device: torch.device, dtype: torch.dtype) -> float | None:
    """Tensor-core MMA peak in TFLOP/s: `MMA_WAVES` CTAs per SM each running
    `MMA_ITERS` x 2 independent register-resident BM x BN x BK `tl.dot`s into fp32
    accumulators; None without Triton or where the dtype does not compile."""
    if triton is None or device.type != "cuda":
        return None
    bm, bn, bk = MMA_TILE
    grid = torch.cuda.get_device_properties(device).multi_processor_count * MMA_WAVES
    a = torch.randn(bm, bk, device=device).to(dtype)
    b = torch.randn(2 * bk, bn, device=device).to(dtype)
    out = torch.empty(grid, bm, bn, device=device)

    def run():
        _mma_loop[(grid,)](a, b, out, MMA_ITERS, bm, bn, bk, num_warps=MMA_WARPS)

    try:
        ms = _timed_ms(run, device, 3)
    except Exception:  # noqa: BLE001
        return None
    return 2 * 2 * bm * bn * bk * MMA_ITERS * grid / (ms * 1e-3) / 1e12


def link_gbps(device: torch.device, pinned: bool) -> dict[str, float]:
    """GB/s (1e9) of host -> device and device -> host copies of `LINK_BYTES`."""
    host = torch.empty(LINK_BYTES, dtype=torch.uint8, pin_memory=pinned)
    dev = torch.empty(LINK_BYTES, dtype=torch.uint8, device=device)
    h2d = _timed_ms(lambda: dev.copy_(host, non_blocking=pinned), device, REPEATS // 2)
    d2h = _timed_ms(lambda: host.copy_(dev, non_blocking=pinned), device, REPEATS // 2)
    return {
        "h2d_gbps": LINK_BYTES / (h2d * 1e-3) / 1e9,
        "d2h_gbps": LINK_BYTES / (d2h * 1e-3) / 1e9,
    }


def probe(device: torch.device) -> dict:
    mod = _accel(device)
    props = mod.get_device_properties(device)
    out = {"name": props.name, "total_gib": props.total_memory / 1024**3}
    bf16 = device.type == "xpu" or torch.cuda.get_device_capability(device) >= (8, 0)
    dtypes = {"fp16": torch.float16, **({"bf16": torch.bfloat16} if bf16 else {})}
    out["gemm_tflops"] = {k: gemm_tflops(device, v) for k, v in dtypes.items()}
    fp8 = device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)
    if fp8:
        out["gemm_tflops"]["fp8"] = fp8_tflops(device)
        dtypes["fp8"] = torch.float8_e4m3fn
    mma = {k: mma_tflops(device, v) for k, v in dtypes.items()}
    out["mma_tflops"] = {k: v for k, v in mma.items() if v is not None}
    out["link"] = {
        "pinned": link_gbps(device, True),
        "pageable": link_gbps(device, False),
    }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="")
    args = ap.parse_args(argv)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "xpu:0")
    )
    Path(args.out).write_text(json.dumps(probe(device), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
