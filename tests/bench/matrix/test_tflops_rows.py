"""The per-kernel TFLOP/s of an ncu report (FLOP per HMMA instruction from the kernel's
mma shape, peak at each launch's clock, FLOP-weighted aggregation, the tensor-op
cross-check) and the clock window of the GEMM ceiling probe."""

from datetime import datetime

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.measure import ceiling, tflops
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
INST, OPS = tflops.INST[86], tflops.OPS[86]
CSV = "\n".join(
    [
        "==PROF== Disconnected from process 1",
        f'"ID","Kernel Name","{tflops.DURATION}","{tflops.CLOCK}","{INST}","{OPS}",'
        '"launch__grid_size","launch__block_size"',
        '"","","nsecond","cycle/second","inst","","",""',
        '"0","ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_tn",'
        '"1000000","1750000000","15000000","61440000000","528","256"',
        '"1","void pytorch_flash::flash_bwd_dq_dk_dv_loop_kernel<T>(Params)",'
        '"2000000","1750000000","25000000","102400000000","96","256"',
    ]
)


def test_flop_per_instruction_follows_the_mma_shape():
    assert tflops.flop_per_inst("ampere_fp16_s16816gemm_fp16_256x128_tn", 86) == 4096
    assert tflops.flop_per_inst("turing_fp16_s1688gemm_fp16_256x128_tn", 75) == 2048
    wmma = "cutlass_75_wmma_tensorop_s161616gemm_f16_16x16_128x2_nt"
    assert tflops.flop_per_inst(wmma, 75) == 2048
    assert tflops.flop_per_inst(wmma, 86) == 4096
    assert tflops.flop_per_inst(wmma, 70, override=512) == 512
    with pytest.raises(ValueError, match="sm70"):
        tflops.flop_per_inst(wmma, 70)


def _summary(flop_override: int | None = None) -> dict:
    rows = tflops.launches(CSV, 86, 82, INST, OPS, 512, flop_override)
    return tflops.aggregate(rows)


def check_summary(summ: dict) -> None:
    peak = 82 * 512 * 1.75e9 / 1e12
    by = {k["kernel"]: k for k in summ["kernels"]}
    gemm = by["ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_tn"]
    assert gemm["tflops"] == pytest.approx(61.44)
    assert gemm["pct_peak"] == pytest.approx(61.44 / peak * 100)
    assert gemm["kind"] == "gemm" and gemm["share_pct"] == pytest.approx(100 / 3)
    flash = by["void pytorch_flash::flash_bwd_dq_dk_dv_loop_kernel<T>"]
    assert flash["kind"] == "attention-bwd" and flash["tflops"] == pytest.approx(51.2)
    total = next(t for t in summ["totals"] if t["kind"] == "all")
    assert total["tflops"] == pytest.approx(163.84 / 3)
    assert total["clock_ghz"] == pytest.approx(1.75)
    assert summ["ops_ratio"]["median"] == pytest.approx(1.0)


def test_report_rows_and_totals():
    check_summary(_summary())
    must_fail(lambda: check_summary(_summary(2048)), "half the FLOP per instruction")
    assert "| gemm |" in tflops.table(_summary(), 12)
    assert tflops.aggregate([])["kernels"] == []


def _stamp(second: int) -> str:
    return f"2026/10/19 13:00:{second:02d}.000"


def check_window(samples: list[list[float]]) -> None:
    assert samples == [[1740.0, 368.0, 61.0], [1755.0, 369.0, 62.0]], samples


def test_ceiling_clock_window():
    lines = [
        f"{_stamp(1)}, 210, 30.5, 40",
        f"{_stamp(5)}, 1740, 368.0, 61",
        f"{_stamp(6)}, [N/A], 368.0, 61",
        f"{_stamp(7)}, 1755, 369.0, 62",
        "garbage",
    ]
    lo, hi = datetime(2026, 10, 19, 13, 0, 4), datetime(2026, 10, 19, 13, 0, 8)
    check_window(ceiling.window_samples(lines, lo, hi))
    early = datetime(2026, 10, 19, 13, 0, 0)
    must_fail(
        lambda: check_window(ceiling.window_samples(lines, early, hi)), "idle sample"
    )
