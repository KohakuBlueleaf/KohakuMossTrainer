# Rotated int8 linear (ConvRot)

Measured by `scripts/bench/kernel/rotate.py` (with `quant.py` for the plain int8
linear), one JSON per card under `runs/kernel/`. The design is in
[quant-kernels.md](../internals/quant-kernels.md#rotated-int8-linear).

Columns:

* **rotation**: `group_hadamard` alone on x [M, K] (g = 256), on the fastest backend;
* **int8 fwd** / **rotated fwd**: the registered `quant_linear` and
  `quant_linear_rotated` forward as the tuner selected them;
* **fwd x** / **fwd+bwd x**: rotated over plain (forward, then forward + input
  gradient);
* **separate / best fused**: the tuner's own device times of the rotated forward and
  dgrad candidates.

All times are device ms (profiler kernel time); fp16 rows match the bf16 ones within
a few percent.

| card | status |
| --- | --- |
| RTX 3090 (sm86) | measured below |
| RTX 4090 (sm89) | pending: no sm89 card in the pool |
| V100 (sm70) | pending: the host is unreachable |

## RTX 3090 (sm86), bf16

| layer (N x K) | M | rotation | int8 fwd | rotated fwd | fwd x | fwd+bwd x | fwd: separate / best fused | dgrad: separate / best fused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qkv (9216 x 3072) | 4096 | 0.062 | 3.147 | 3.215 | 1.022 | 1.078 | 3.299 / 6.908 | 3.333 / 3.986 |
| qkv (9216 x 3072) | 8192 | 0.123 | 5.913 | 6.097 | 1.031 | 1.034 | 6.245 / 13.054 | 6.256 / 8.072 |
| qkv (9216 x 3072) | 16384 | 0.243 | 11.806 | 12.151 | 1.029 | 1.022 | 12.283 / 26.112 | 12.782 / 15.558 |
| o (3072 x 3072) | 4096 | 0.062 | 1.057 | 1.115 | 1.055 | 1.139 | 1.179 / 2.239 | 1.218 / 1.449 |
| o (3072 x 3072) | 8192 | 0.123 | 1.992 | 2.114 | 1.061 | 1.116 | 2.167 / 4.416 | 2.308 / 2.833 |
| o (3072 x 3072) | 16384 | 0.242 | 3.988 | 4.260 | 1.068 | 1.100 | 4.285 / 8.680 | 4.330 / 5.491 |
| up_gate (16384 x 3072) | 4096 | 0.061 | 5.249 | 5.346 | 1.019 | 1.040 | 5.447 / 11.632 | 5.841 / 7.234 |
| up_gate (16384 x 3072) | 8192 | 0.123 | 10.686 | 10.840 | 1.014 | 1.013 | 10.914 / 23.254 | 10.985 / 13.989 |
| up_gate (16384 x 3072) | 16384 | 0.243 | 21.103 | 21.413 | 1.015 | 1.015 | 21.498 / 46.478 | 23.099 / 28.434 |
| down (3072 x 8192) | 4096 | 0.174 | 2.797 | 2.957 | 1.057 | 1.094 | 3.052 / 5.824 | 3.042 / 3.689 |
| down (3072 x 8192) | 8192 | 0.327 | 5.305 | 5.626 | 1.061 | 1.086 | 5.771 / 11.606 | 5.724 / 7.353 |
| down (3072 x 8192) | 16384 | 0.649 | 10.474 | 11.143 | 1.064 | 1.069 | 11.226 / 22.837 | 11.403 / 14.454 |

* The rotation runs at about 830 GB/s (read + write of x), so its share is the bytes of
  x over the GEMM time: largest for the small `o` GEMM and the K = 8192 `down` input.
* The fused forward rotates each x tile once per output tile (N / BN times) on the
  CUDA cores and runs about 2x the separate pass; the fused dgrad (one rotation per
  output tile, but a fixed 256-wide output tile) runs 1.2-1.3x. The tuner selects
  `separate` on this card at every shape.
* The plain int8 linear is at par with compiled dequantize + matmul on this card. Per
  shape the tuner picks either the Triton tile or the compiled path.
  * Forward: the tile wins mostly at 4096 tokens, the compiled path at 16384 tokens.
  * Dgrad: the compiled path wins at every shape.
  * The selected path is within 3% of the faster PyTorch path in every row but one
    (bf16 3072 x 3072 at 4096 tokens).
