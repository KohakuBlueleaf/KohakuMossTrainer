# Varlen attention

The packed variable-length attention of each arch against PyTorch, forward + backward
and forward only, then its precision against fp64. Forward + backward, the bound entry
beats SDPA on every measured card and case; on the long single-segment shapes it binds
torch's own `varlen_attn` where the build has it
([The single-segment ceiling](#the-single-segment-ceiling)).

## Columns

| column | meaning |
| --- | --- |
| ours | the registered `attention_varlen` of the arch (bench label `kohakumoss_<backend>_chosen`): per problem key, the backend's kernels (TileLang on sm70 / sm75, Triton on sm80+) timed forward + backward against torch's `varlen_attn` where the torch build has it, else the per-segment SDPA loop, and the faster bound |
| kernels | the backend's kernels alone (`kohakumoss_<backend>_kernels`) |
| SDPA | PyTorch's per-segment `scaled_dot_product_attention` (flash on sm80+ Linux, memory-efficient on sm70 / sm75 and on Windows) |
| varlen_attn | torch's FA2-based varlen op (Linux sm80+ builds only) |

Among the implementations whose precision stats match PyTorch's (see
[Precision](#precision)), the entry binds the fastest per shape. Each bench row records
what a chosen entry bound (`bound`) and the error statistics of o, dq, dk, dv against
fp64 (`stats`).

## Method

- Bench: `scripts/bench/kernel/attention.py`, 24 heads x 128.
- Cases: packed segments, each the image tokens of a 1 MP / 2 MP / 512 px MOSS image
  plus its caption; the text-refiner case is caption only.
- Time: device time per call, best of 10 (CUDA events). fwd + bwd is the median over
  interleaved rounds (`--rounds`; each round times every implementation in turn, best of
  the calls).
- Memory: the peak above the allocation at the start of the case.
- Errors: computed on 2 of the heads against an fp64 run.
- The padded SDPA baseline reads the segment bounds on the host inside its timed region,
  as that PyTorch path does in use.

## fwd + bwd, ms

| case | V100 ours | SDPA | 2080 Ti ours | SDPA | 3090 ours | kernels | SDPA | 5090 ours | kernels | SDPA | 4090 Windows ours | SDPA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 MP x2 | **45.6** | 74.2 | **70.8** | 78.3 | **23.7** | 26.8 | 26.3 | **7.49** | 8.37 | 8.08 | **13.0** | 26.9 |
| 2 MP x1 | **96.8** | 137.1 | **138.7** | 150.3 | **44.4** | 49.7 | 45.1 | **14.36** | 15.85 | 14.45 | | |
| packed mixed | **33.6** | 56.8 | **51.6** | 61.0 | **17.3** | 19.3 | 20.5 | **5.58** | 6.19 | 6.50 | **9.2** | 20.9 |
| 512 px x8 | **13.3** | 25.4 | **19.8** | 32.7 | **6.96** | 7.77 | 15.4 | **2.68** | 2.67 | 5.38 | | |
| text refiner | **1.35** | 4.35 | **1.93** | 4.52 | **1.16** | 1.43 | 5.07 | **0.54** | 0.65 | 2.33 | **0.66** | 2.39 |

fp16 on V100, 2080 Ti and 4090; bf16 on the 3090 and the 5090 (fp16 rows are within 2 %
of bf16). The 3090 and 5090 columns are medians of interleaved rounds (`--rounds 8` /
`--rounds 6`: each round times every implementation in turn, best of 10 calls), the
other columns single best-of-10 runs.

Speed-up of ours over SDPA, fwd + bwd:

| case | V100 | 2080 Ti | 3090 | 5090 | 4090 Windows |
| --- | --- | --- | --- | --- | --- |
| 1 MP x2 | 1.63x | 1.11x | 1.11x | 1.08x | 2.07x |
| 2 MP x1 | 1.42x | 1.08x | 1.02x | 1.01x | |
| packed mixed | 1.69x | 1.18x | 1.19x | 1.16x | 2.27x |
| 512 px x8 | 1.91x | 1.65x | 2.21x | 2.01x | |
| text refiner | 3.22x | 2.34x | 4.39x | 4.30x | 3.62x |

The entry times each candidate once per problem key and binds the faster, so candidates
within the timing noise of each other bind either way between runs; in the run above
the 5090's 512 px x8 key bound our Triton kernels (2.68 ms, 2.0x SDPA). The table has
no fwd + bwd column for torch's `varlen_attn`, so its time on that key is not compared
here.

## fwd only, ms

| case | V100 ours | SDPA | 2080 Ti ours | SDPA | 3090 kernels | SDPA | varlen_attn | 5090 kernels | SDPA | varlen_attn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 MP x2 | **9.9** | 14.4 | **15.4** | 16.2 | **6.62** | 7.57 | 6.62 | 2.16 | 2.16 | 2.13 |
| 2 MP x1 | **20.6** | 29.1 | **29.7** | 32.1 | **12.38** | 12.78 | 12.64 | **4.12** | 4.15 | 4.11 |
| packed mixed | **7.3** | 11.4 | **11.2** | 12.4 | **4.73** | 5.27 | 4.92 | **1.62** | 1.64 | 1.63 |
| 512 px x8 | **2.91** | 4.19 | **4.37** | 4.86 | **1.86** | 2.47 | 1.84 | **0.75** | 1.02 | 0.69 |
| text refiner | **0.33** | 0.63 | **0.45** | 0.60 | **0.24** (TileLang) | 0.30 | 0.20 | **0.12** (TileLang) | 0.48 | 0.14 |

## The single-segment ceiling

With one long segment the per-segment SDPA is a single FA2 call, and FA2 runs at the
card's measured 16-bit MMA rate (fp32 accumulation; `hardware.device.measure_rates`):

| card | MMA probe | FA2 fwd (2 MP x1) | FA2 bwd (2 MP x1) | our fwd | our fused bwd |
| --- | --- | --- | --- | --- | --- |
| RTX 3090 | 68 TFLOP/s | 75 TFLOP/s | 74 TFLOP/s | 76 TFLOP/s | 63 TFLOP/s |
| RTX 5090 | 233 TFLOP/s | 228 TFLOP/s | 232 TFLOP/s | 235 TFLOP/s | 200 TFLOP/s |

(fwd = 2 GEMMs, bwd = 5 GEMMs of `2 N^2 D H` flops each.) There is no headroom left on
those shapes for any fp32-accumulating kernel, so the entry binds torch's `varlen_attn`
there and is 0.6 - 1.6 % faster than SDPA fwd + bwd on 2 MP x1 (1.01x / 1.02x in the
speed-up table; it saves SDPA's layout transposes). Our
forward reaches the same rate; our backward stays at 85 - 86 % of FA2's on both cards;
see [kernel-dev.md](../internals/kernel-dev.md#measured-design-per-arch) for the tile
study. On every packed or short-segment shape our kernels (or the `varlen_attn` the
entry binds) beat SDPA fwd + bwd by 1.08 - 4.39x (the speed-up table), because one
launch covers every segment.

## Precision

Both results against fp64 (cosine; |out - ref|; |out - ref| / |ref| over nonzero ref),
RTX 5090, ours = the registered entry, SDPA = PyTorch's flash. Every attention matmul,
ours and PyTorch's, takes 16-bit operands with fp32 accumulation
([kernel-dev.md](../internals/kernel-dev.md#matmul-precision-and-error-statistics)).
Mins are all zero or below 1e-8 and omitted.

bf16, 1 MP x2 (2 heads):

| | tensor | cos | abs mean | abs median | abs max | rel mean | rel median | rel max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SDPA | o | 0.9999976 | 4.22e-5 | 3.37e-5 | 4.39e-4 | 1.80e-2 | 2.17e-3 | 5.7e3 |
| ours | o | 0.9999974 | 4.37e-5 | 3.51e-5 | 4.39e-4 | 1.85e-2 | 2.24e-3 | 5.7e3 |
| SDPA | dq | 0.9999972 | 4.53e-5 | 3.63e-5 | 9.21e-4 | 1.79e-2 | 2.34e-3 | 4.5e3 |
| ours | dq | 0.9999972 | 4.53e-5 | 3.63e-5 | 9.21e-4 | 1.77e-2 | 2.34e-3 | 4.6e3 |
| SDPA | dk | 0.9999972 | 4.48e-5 | 3.55e-5 | 7.19e-4 | 1.60e-2 | 2.32e-3 | 3.4e3 |
| ours | dk | 0.9999972 | 4.48e-5 | 3.55e-5 | 7.19e-4 | 1.62e-2 | 2.32e-3 | 3.1e3 |
| SDPA | dv | 0.9999972 | 4.47e-5 | 3.58e-5 | 6.68e-4 | 3.06e-2 | 2.32e-3 | 1.2e4 |
| ours | dv | 0.9999972 | 4.47e-5 | 3.58e-5 | 6.68e-4 | 3.05e-2 | 2.32e-3 | 1.2e4 |

fp16, 1 MP x2: every stat of dq, dk, dv within 1 % of SDPA's (cosine 1.0000000); o abs
mean 5.46e-6 vs 5.28e-6, rel mean 1.96e-3 vs 2.19e-3, rel max 397 vs 790.

Worse than SDPA, with the reason: o's mean / median errors are 3 - 4 % above flash's in
both dtypes (bf16 abs mean 4.37e-5 vs 4.22e-5). The two kernels round P to 16 bits from
differently formed exponents (ours `exp2((s - m) c)` with the raw row max, flash
`exp2(s c - m c)`); cosine is equal to 7 digits. dq / dk / dv's rel max differs by up
to 2x either way (one element each, over near-zero reference values).

Clustered queries and keys (logits ~1e5), bf16, the test's first draw: dQ cosine 0.988
(ours) vs 0.974 (SDPA), abs mean 7.5e-3 vs 1.06e-2; dK cosine 0.99996 vs 0.99988; dV
0.999998 vs 0.99999. The relative stats there are dominated by fp64 reference values
near 1e-190 (see kernel-dev.md) and are not compared.

XPU runs the same kernels. On every device the tests' PyTorch baseline is plain
per-segment SDPA in the 16-bit dtype on whichever SDPA backend the device selects
(`pytorch_sdpa`, see kernel-dev.md), with fp64 as the reference; the SDPA rows above
are flash SDPA on the RTX 5090.
