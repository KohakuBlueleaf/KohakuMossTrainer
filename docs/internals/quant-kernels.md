# Quantized frozen-base kernels

`kohakumoss_kernels.quant` serves `quant_linear(x, q, scale, bias, fmt)`, the
op `kohakumoss.quant.QuantWeight.linear` resolves once per weight (`bind`).

## Storage

`q` is `[N, K]` (`int8`, or `float8_e4m3fn` for `fmt="fp8"`), `scale` is an
fp32 per-row vector `[N]`, the weight is `q * scale[:, None]`. This is the
layout `QuantWeight.from_weight` produces (int8: `round(w / s)`,
`s = max|w_row| / 127`; fp8: `s = max|w_row| / 448`).

## What runs

| arch | int8 | fp8 |
| --- | --- | --- |
| sm70 / sm75 | TileLang weight-only GEMM (fp16) | — |
| sm80 / sm86 | Triton weight-only GEMM (fp16 / bf16) | — |
| sm89 / sm120 | Triton weight-only GEMM (fp16 / bf16) | Triton |
| CPU, XPU | eager reference | eager reference |

* **Forward** `y = (x @ q^T) * scale + bias`: the int8 / e4m3 tile is
  converted to the activation dtype in registers or shared memory (exact for
  both formats), multiplied on the MMA units with an fp32 accumulator, and the
  scale and bias are applied to that accumulator before one rounding at the
  store.
* **Input gradient** `dx = dy @ (q * scale[:, None])`: the scaled tile is
  rounded once to the activation dtype — the same rounding the
  dequantize-then-matmul reference applies — and accumulated in fp32.
* **No weight gradient** (the base is frozen); a trainable bias gets the fp32
  column sum of `dy`.

## Selection

Per (device, token bucket, N, K, dtypes, bias) the tuner times every kernel
tile config **and** the two PyTorch paths — dequantize + `F.linear` eager, and
the same under `torch.compile` — and keeps the fastest. A kernel therefore
only runs where it beats both PyTorch paths on that card; elsewhere the op is
the PyTorch path. The choice is resolved once per key and held in a dict, so
the per-call cost is one lookup. Only a settled choice (`tuning.decided`) is
held: a first call where the tuner may not time — inside a trace, or the
forward GEMM inside a checkpoint recompute — runs the planner's first
candidate for that call alone, and the next call that may time measures the
key. The dgrad GEMM runs inside the autograd backward and launches no autograd
itself, so it is timed there (`in_backward=True`).

The op is a `torch.library` custom op (`kohakumoss::quant_fwd` /
`kohakumoss::quant_dgrad`) with fake implementations and registered autograd,
so it stays in a `torch.compile(fullgraph=True)` graph.

## Checks

* `tests/kernels/test_quant.py` — forward, input and bias gradients against an
  fp64 reference in the ULP bar, off-tile shapes, fullgraph compile, and a
  scale-on-the-wrong-axis variant the bar must reject.
* `scripts/bench/kernel/quant.py` — kernel time of every backend against eager
  and compiled dequantize+matmul at the DiT layer shapes, with the path the
  tuner selected for each case.
