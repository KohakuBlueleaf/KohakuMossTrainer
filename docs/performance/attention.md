# Varlen attention

`scripts/bench/kernel/attention.py`, 24 heads x 128, packed segments (image tokens of
a 1 MP / 2 MP / 512 px MOSS image plus its caption; the text-refiner case is caption
only). Device time per call, best of 10 (CUDA events), fp16. "SDPA" is PyTorch's
per-segment `scaled_dot_product_attention` (flash on sm80+ Linux, memory-efficient
elsewhere); `varlen_attn` is torch's FA2-based varlen op (Linux sm80+ builds only).

The registered `attention_varlen` on sm80+ races, per problem key, our Triton kernels,
our TileLang kernels and (where the torch build has it, and not under
`KOHAKUMOSS_DETERMINISTIC=1`) torch's `varlen_attn`, on forward + backward; "raced"
is that entry, "kernels" is our Triton kernels alone.

## fwd + bwd, ms

| case | V100 (sm70) TileLang | SDPA | 2080 Ti (sm75) TileLang | SDPA | 3090 (sm86) raced | 3090 kernels | SDPA (flash) | varlen_attn | 4090 Windows (sm89) raced | SDPA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 MP x2 | **45.6** | 74.2 | **70.8** | 78.3 | **24.0** | 27.3 | 26.2 | 23.8 | **13.0** | 26.9 |
| 2 MP x1 | **96.8** | 137.1 | **138.7** | 150.3 | **45.3** | 50.4 | 45.6 | 45.1 | | |
| packed mixed | **33.6** | 56.8 | **51.6** | 61.0 | **17.4** | 19.5 | 20.7 | 17.3 | **9.2** | 20.9 |
| 512 px x8 | **13.3** | 25.4 | **19.8** | 32.7 | **7.0** | 7.8 | 15.5 | 7.0 | | |
| text refiner | **1.35** | 4.35 | **1.93** | 4.52 | **0.91** | 1.26 | 4.36 | 1.12 | **0.66** | 2.39 |

## fwd only, ms

| case | V100 TileLang | SDPA | 2080 Ti TileLang | SDPA | 3090 kernels | SDPA (flash) | varlen_attn | 4090 Windows kernels | SDPA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 MP x2 | **9.9** | 14.4 | **15.4** | 16.2 | **6.79** | 7.13 | 6.73 | **3.16** | 4.67 |
| 2 MP x1 | **20.6** | 29.1 | **29.7** | 32.1 | **12.6** | 12.9 | 12.8 | | |
| packed mixed | **7.3** | 11.4 | **11.2** | 12.4 | **4.79** | 5.33 | 4.97 | **2.23** | 3.54 |
| 512 px x8 | **2.91** | 4.19 | **4.37** | 4.86 | **1.88** | 2.32 | 1.87 | | |
| text refiner | **0.33** | 0.63 | **0.45** | 0.60 | **0.25** (TileLang) | 0.49 | 0.25 | **0.13** (TileLang) | 0.39 |

bf16 on the 3090 and the 4090 matches the fp16 timings within 2 %.

## Measured gap: our kernels alone on sm86 Linux

Our Triton forward beats SDPA flash on every case. Our backward alone trails FA2's on
long single segments: fwd + bwd 27.3 vs 26.2 ms (1 MP x2) and 50.4 vs 45.6 ms (2 MP x1),
because the fused backward (dQ through fp32 atomics) reaches about 95 % of FA2's
backward throughput (19.9 vs about 18.4 ms on 1 MP x2) and the deterministic split
backward recomputes two extra GEMMs. On those shapes the race binds torch's
`varlen_attn`, whose error equals SDPA flash (and ours) at these shapes; everywhere
else, and on every arch without flash in the torch build (sm70, sm75, Windows), our
kernels are the fastest entry.

## Precision (relative L2 vs fp64, fp16, 1 MP x2)

| | o | dQ | dK | dV |
| --- | --- | --- | --- | --- |
| ours (all archs) | 2.82e-4 - 2.87e-4 | 2.95e-4 - 2.96e-4 | 2.93e-4 | 2.93e-4 - 2.97e-4 |
| SDPA memory-efficient (sm70 / sm75 / Windows) | 2.86e-4 - 2.87e-4 | 3.64e-4 | 3.61e-4 | 2.93e-4 - 2.97e-4 |
| SDPA flash / varlen_attn (sm86 Linux) | 2.85e-4 | 2.96e-4 | 2.93e-4 | 2.96e-4 |

With clustered queries and keys (logits around 1e5) our kernels keep o / dV cosine
above 0.9999 and dQ / dK above 0.995 against fp64, or at least PyTorch's
(`tests/kernels/test_attention.py`).
