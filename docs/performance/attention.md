# Varlen attention

`scripts/bench/kernel/attention.py`, 24 heads x 128, packed segments (image tokens of
a 1 MP / 2 MP / 512 px MOSS image plus its caption; the text-refiner case is caption
only). Device time per call, best of 10 (CUDA events). "SDPA" is PyTorch's per-segment
`scaled_dot_product_attention` (flash on sm80+ Linux, memory-efficient on sm70 / sm75
and on Windows); `varlen_attn` is torch's FA2-based varlen op (Linux sm80+ builds only).

"ours" is the registered `attention_varlen` of the arch: TileLang on sm70 / sm75; on
sm80+ the per-key race of our Triton kernels, our TileLang kernels and (where the torch
build has it, and not under `KOHAKUMOSS_DETERMINISTIC=1`) torch's `varlen_attn`, timed on
forward + backward. "kernels" is our Triton kernels alone (p16 precision mode).

## fwd + bwd, ms

| case | V100 ours | SDPA | 2080 Ti ours | SDPA | 3090 ours | kernels | SDPA | 5090 ours | kernels | SDPA | 4090 Windows ours | SDPA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 MP x2 | **45.6** | 74.2 | **70.8** | 78.3 | **24.0** | 27.3 | 26.2 | **7.50** | 8.38 | 8.10 | **13.0** | 26.9 |
| 2 MP x1 | **96.8** | 137.1 | **138.7** | 150.3 | **45.3** | 50.4 | 45.6 | **14.41** | 15.89 | 14.52 | | |
| packed mixed | **33.6** | 56.8 | **51.6** | 61.0 | **17.4** | 19.5 | 20.7 | **5.61** | 6.21 | 6.51 | **9.2** | 20.9 |
| 512 px x8 | **13.3** | 25.4 | **19.8** | 32.7 | **7.0** | 7.8 | 15.5 | **2.39** | 2.72 | 5.40 | | |
| text refiner | **1.35** | 4.35 | **1.93** | 4.52 | **0.91** | 1.26 | 4.36 | **0.58** | 0.76 | 2.33 | **0.66** | 2.39 |

fp16 on V100, 2080 Ti, 3090 and 4090; bf16 on the 5090 (its fp16 rows are within 2 % of
bf16, as are the 3090's and 4090's bf16 rows of fp16). The 5090 columns are medians of
12 interleaved rounds (`--rounds 12`: each round times every implementation in turn,
best of 10 calls), the other columns single best-of-10 runs.

Speed-up of ours over SDPA, fwd + bwd:

| case | V100 | 2080 Ti | 3090 | 5090 | 4090 Windows |
| --- | --- | --- | --- | --- | --- |
| 1 MP x2 | 1.63x | 1.11x | 1.09x | 1.08x | 2.07x |
| 2 MP x1 | 1.42x | 1.08x | 1.01x | 1.01x | |
| packed mixed | 1.69x | 1.18x | 1.19x | 1.16x | 2.27x |
| 512 px x8 | 1.91x | 1.65x | 2.20x | 2.26x | |
| text refiner | 3.22x | 2.34x | 4.81x | 4.03x | 3.62x |

## fwd only, ms

| case | V100 ours | SDPA | 2080 Ti ours | SDPA | 3090 kernels | SDPA | varlen_attn | 5090 kernels | SDPA | varlen_attn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 MP x2 | **9.9** | 14.4 | **15.4** | 16.2 | **6.79** | 7.13 | 6.73 | 2.16 | 2.16 | 2.13 |
| 2 MP x1 | **20.6** | 29.1 | **29.7** | 32.1 | **12.6** | 12.9 | 12.8 | **4.12** | 4.15 | 4.11 |
| packed mixed | **7.3** | 11.4 | **11.2** | 12.4 | **4.79** | 5.33 | 4.97 | **1.62** | 1.64 | 1.63 |
| 512 px x8 | **2.91** | 4.19 | **4.37** | 4.86 | **1.88** | 2.32 | 1.87 | **0.75** | 1.02 | 0.69 |
| text refiner | **0.33** | 0.63 | **0.45** | 0.60 | **0.25** (TileLang) | 0.49 | 0.25 | **0.12** (TileLang) | 0.48 | 0.14 |

## The single-segment ceiling

With one long segment the per-segment SDPA is a single FA2 call, and FA2 runs at the
card's measured 16-bit MMA rate (fp32 accumulation; `hardware.device.probe`):

| card | MMA probe | FA2 fwd (2 MP x1) | FA2 bwd (2 MP x1) | our fwd | our fused bwd |
| --- | --- | --- | --- | --- | --- |
| RTX 3090 | 68 TFLOP/s | 72 TFLOP/s | 71 TFLOP/s | 74 TFLOP/s | 61 TFLOP/s |
| RTX 5090 | 233 TFLOP/s | 228 TFLOP/s | 232 TFLOP/s | 235 TFLOP/s | 200 TFLOP/s |

(fwd = 2 GEMMs, bwd = 5 GEMMs of `2 N^2 D H` flops each.) There is no headroom left on
those shapes for any fp32-accumulating kernel, so the race binds torch's `varlen_attn`
there and the entry is 0.7 - 1 % faster than SDPA (it saves SDPA's layout transposes).
Our forward reaches the same rate; our backward stays at 86 % of FA2's on both cards:
see
[kernel-dev.md](../internals/kernel-dev.md#measured-design-per-arch) for the tile study.
On every packed or short-segment shape our kernels (or `varlen_attn` inside the race)
beat SDPA by 1.1 - 4.8x, because one launch covers every segment.

## Precision (relative L2 vs fp64, fp16, 1 MP x2)

| | o | dQ | dK | dV |
| --- | --- | --- | --- | --- |
| ours, p16 (all archs) | 2.82e-4 - 2.87e-4 | 2.95e-4 - 2.96e-4 | 2.93e-4 | 2.92e-4 - 2.97e-4 |
| ours, split (Triton) | 2.07e-4 | 2.95e-4 | 2.93e-4 | 2.92e-4 |
| SDPA memory-efficient (sm70 / sm75 / Windows) | 2.86e-4 - 2.87e-4 | 3.64e-4 | 3.61e-4 | 2.93e-4 - 2.97e-4 |
| SDPA flash / varlen_attn (sm86 / sm120 Linux) | 2.83e-4 - 2.85e-4 | 2.96e-4 | 2.93e-4 | 2.92e-4 - 2.96e-4 |

In ULP of the dtype (sm120, 8 draws of the test case, max / RMS): flash SDPA o 44 / 0.26,
dQ 134 / 0.31; our p16 kernels the same within 2 %; `split` 0.52 / 0.20 (correctly
rounded). The precision mode and its gate are in
[kernel-dev.md](../internals/kernel-dev.md#precision-modes-and-the-precision-gate).

With clustered queries and keys (logits around 1e5; the test's first draw, sm120 bf16),
dQ's cosine to fp64 is 0.974 for flash SDPA, 0.988 for our p16 kernels and 0.999997 for
`split`; dK and dV are above 0.9999 for all three.
