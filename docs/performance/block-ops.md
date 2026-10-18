# Block ops: kernels vs PyTorch per GPU

Forward + backward device time in ms of the MOSS block ops (norm, residual, qk-norm +
rope, SiTU, loss, fused linears) per GPU: each kernel against `torch.compile` and the
eager composition, and the candidate the per-shape choice binds.

| column | meaning |
| --- | --- |
| TileLang, Triton | the backend's kernel |
| compile | `torch.compile` of the eager reference with dynamic shapes (the compile candidate) |
| eager | the eager composition |
| bound | the candidate the per-shape choice binds; `kernel` is the arch's first backend, "a / b" means either binds between runs |

- **Method:** best of 20, L2 flushed (`scripts/bench/kernel/`,
  `kohakumoss_kernels/bench/blockops.py`); MOSS shapes (D = 3072, FFN 2 x 8192, 24 heads
  x 128).
- **Choice:** the per-shape choice times every candidate per problem and binds the
  fastest, so an op never runs a candidate slower than the fastest one it measured; rows
  whose candidates are within a few percent bind either way between runs.
- **Precision:** every listed kernel passes the error-statistics check against PyTorch
  ([fp16.md](../internals/fp16.md#eager-reference-and-error-statistics)) in every row.

## sm75 (Tesla T10), fp16

| op | case | TileLang | Triton | compile | eager | bound |
| --- | --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.512 | 0.425 | 0.553 | 6.73 | kernel / triton |
| rms_norm_mod | N=4400, all grads | 0.680 | 0.452 | 1.319 | 10.18 | triton |
| rms_norm | N=4400 | 0.651 | 0.442 | 1.084 | 5.40 | kernel / triton |
| rms_norm_mod | N=16400 B=4, dx | 1.830 | 1.423 | 2.332 | 24.98 | triton |
| rms_norm_mod | N=16400 B=4, all grads | 1.614 | 1.430 | 5.114 | 43.75 | triton |
| rms_norm | N=16400 | 1.651 | 1.473 | 3.837 | 19.92 | triton |
| gated_residual | N=4400, gate frozen | 0.524 | 0.404 | 0.451 | 3.31 | triton / compile |
| gated_residual | N=4400, gate trained | 0.680 | 0.482 | 0.561 | 5.18 | triton |
| gated_residual | N=16400 B=4, gate frozen | 1.950 | 1.425 | 1.669 | 10.65 | triton |
| gated_residual | N=16400 B=4, gate trained | 2.287 | 1.698 | 1.947 | 16.58 | triton |
| qk_norm_rope | N=4396, weights frozen | 4.110 | 0.765 | 3.692 | 15.40 | triton |
| qk_norm_rope | N=4396, weights trained | 4.157 | 0.858 | 4.942 | 16.61 | triton |
| qk_norm_rope | N=17381 B=4, weights frozen | 16.02 | 3.228 | 14.42 | 60.44 | triton |
| qk_norm_rope | N=17381 B=4, weights trained | 16.09 | 3.595 | 19.82 | 65.06 | triton |
| situ_glu | N=4400 | 1.591 | - | 2.659 | 25.89 | kernel |
| token_mse | N=4096 | 0.108 | 0.100 | 0.025 | 0.104 | compile |
| token_mse | N=15264 | 0.178 | 0.072 | 0.070 | 0.442 | compile / triton |
| linear_situ | N=17381 | 74.9 | - | 80.3 | 185.6 | kernel |
| linear_gated_residual, FFN down | N=17381 | 36.0 | - | 36.1 | 46.1 | kernel / compile |
| linear_gated_residual, out proj | N=17381 | 14.1 | - | 13.9 | 21.4 | kernel / compile |
| linear_qk_rope | N=17381 | 38.7 | - | 45.0 | 80.8 | kernel |

## sm70 (Tesla V100 32G), fp16

Triton has no column here: its sm70 measurement ran on a GPU another job was using, so
it is not comparable with the other columns and is left out. The per-shape choice of
that run timed Triton among the candidates, so the `bound` column is pending a
re-measure on an idle V100.

| op | case | TileLang | compile | eager | bound |
| --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.489 | 0.486 | 2.59 | pending re-measure |
| rms_norm_mod | N=4400, all grads | 1.190 | 0.660 | 3.37 | pending re-measure |
| rms_norm | N=4400 | 1.977 | 0.578 | 2.13 | pending re-measure |
| rms_norm_mod | N=16400 B=4, dx | 0.991 | 1.196 | 9.30 | pending re-measure |
| rms_norm_mod | N=16400 B=4, all grads | 1.253 | 2.243 | 12.09 | pending re-measure |
| rms_norm | N=16400 | 2.407 | 1.452 | 7.57 | pending re-measure |
| gated_residual | N=4400, gate frozen | 0.733 | 0.516 | 1.14 | pending re-measure |
| gated_residual | N=4400, gate trained | 0.866 | 0.525 | 1.49 | pending re-measure |
| gated_residual | N=16400 B=4, gate frozen | 0.843 | 1.060 | 4.07 | pending re-measure |
| gated_residual | N=16400 B=4, gate trained | 1.064 | 1.087 | 5.36 | pending re-measure |
| qk_norm_rope | N=4396, weights frozen | 2.230 | 2.908 | 6.07 | pending re-measure |
| qk_norm_rope | N=4396, weights trained | 2.279 | 3.344 | 6.69 | pending re-measure |
| qk_norm_rope | N=17381 B=4, weights frozen | 6.932 | 10.96 | 22.99 | pending re-measure |
| qk_norm_rope | N=17381 B=4, weights trained | 7.021 | 12.65 | 24.84 | pending re-measure |
| situ_glu | N=4400 | 0.696 | 1.912 | 9.50 | pending re-measure |
| situ_glu | N=16400 | 2.851 | 6.926 | 35.19 | pending re-measure |
| token_mse | N=4096 | 0.678 | 0.258 | 0.307 | pending re-measure |
| token_mse | N=15264 | 0.666 | 0.346 | 0.389 | pending re-measure |

## sm86 (RTX 3090), Triton, fp16 / bf16

| op | case | Triton fp16 | compile fp16 | Triton bf16 | compile bf16 | eager fp16 | bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.587 | 0.391 | 0.578 | 0.391 | 2.97 | compile |
| rms_norm_mod | N=4400, all grads | 1.158 | 0.554 | 1.232 | 0.588 | 4.72 | compile |
| rms_norm | N=4400 | 1.159 | 0.499 | 1.198 | 0.499 | 2.39 | compile |
| rms_norm_mod | N=16400 B=4, dx | 0.896 | 1.145 | 1.163 | 1.049 | 10.78 | kernel |
| rms_norm_mod | N=16400 B=4, all grads | 1.276 | 1.664 | 1.299 | 1.646 | 17.03 | kernel |
| rms_norm | N=16400 | 1.012 | 1.276 | 1.184 | 1.278 | 8.63 | kernel |
| gated_residual | N=4400, gate frozen | 0.662 | 0.291 | 0.172 | 0.198 | 1.26 | kernel / compile |
| gated_residual | N=4400, gate trained | 0.914 | 0.523 | 0.877 | 0.504 | 2.00 | compile |
| gated_residual | N=16400 B=4, gate frozen | 0.950 | 0.910 | 0.965 | 0.828 | 4.56 | kernel / compile |
| gated_residual | N=16400 B=4, gate trained | 1.025 | 0.836 | 1.019 | 0.979 | 7.50 | kernel / compile |
| qk_norm_rope | N=4396, weights frozen | 0.865 | 1.921 | 0.882 | 2.005 | 6.80 | kernel |
| qk_norm_rope | N=4396, weights trained | 1.073 | 2.332 | 1.230 | 2.343 | 7.46 | kernel |
| qk_norm_rope | N=17381 B=4, weights frozen | 1.402 | 7.188 | 1.399 | 7.415 | 26.06 | kernel |
| qk_norm_rope | N=17381 B=4, weights trained | 1.581 | 8.798 | 1.591 | 8.680 | 28.21 | kernel |
| situ_glu | N=4400 | 0.813 | 1.493 | 0.851 | 1.617 | 11.12 | kernel |
| situ_glu | N=16400 | 2.940 | 5.399 | 2.825 | 5.660 | 41.25 | kernel |
| token_mse | N=4096 | 0.540 | 0.327 | 0.541 | 0.323 | 0.44 | compile |
| token_mse | N=15264 | 0.031 | 0.289 | 0.462 | 0.330 | 0.45 | kernel / compile |

## sm120 (RTX 5090), Triton, fp16 / bf16

| op | case | Triton fp16 | compile fp16 | Triton bf16 | compile bf16 | bound fp16 |
| --- | --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.306 | 0.273 | 0.306 | 0.252 | compile |
| rms_norm_mod | N=4400, all grads | 0.404 | 0.472 | 0.460 | 0.473 | compile |
| rms_norm | N=4400 | 0.393 | 0.242 | 0.440 | 0.284 | compile |
| rms_norm_mod | N=16400 B=4, dx | 0.368 | 0.363 | 0.400 | 0.348 | compile |
| rms_norm_mod | N=16400 B=4, all grads | 0.443 | 0.872 | 0.441 | 0.871 | kernel |
| rms_norm | N=16400 | 0.462 | 0.530 | 0.422 | 0.492 | compile |
| gated_residual | N=4400, gate frozen | 0.305 | 0.250 | 0.325 | 0.235 | compile |
| gated_residual | N=4400, gate trained | 0.440 | 0.311 | 0.436 | 0.306 | compile |
| gated_residual | N=16400 B=4, gate frozen | 0.343 | 0.524 | 0.341 | 0.371 | kernel |
| gated_residual | N=16400 B=4, gate trained | 0.460 | 0.534 | 0.463 | 0.536 | compile |
| qk_norm_rope | N=4396, weights frozen | 0.289 | 0.692 | 0.296 | 0.697 | kernel |
| qk_norm_rope | N=4396, weights trained | 0.419 | 0.855 | 0.425 | 0.812 | kernel |
| qk_norm_rope | N=17381 B=4, weights frozen | 0.662 | 2.520 | 0.663 | 2.593 | kernel |
| qk_norm_rope | N=17381 B=4, weights trained | 0.729 | 3.218 | 0.731 | 3.057 | kernel |
| situ_glu | N=4400 | 0.360 | 0.580 | 0.348 | 0.577 | kernel |
| situ_glu | N=16400 | 1.261 | 1.752 | 1.258 | 1.726 | kernel |
| token_mse | N=4096 | 0.283 | 0.230 | 0.277 | 0.233 | compile |
| token_mse | N=15264 | 0.288 | 0.253 | 0.295 | 0.264 | compile |
| linear_situ | N=17381 | 15.84 | 17.07 | 16.55 | 16.73 | kernel / compile |
| linear_gated_residual, FFN down | N=17381 | 7.80 | 8.01 | 7.65 | 7.86 | kernel |
| linear_gated_residual, out proj | N=17381 | 3.09 | 3.30 | 3.05 | 3.26 | kernel |
| linear_qk_rope | N=17381 | 6.51 | 10.23 | 6.41 | 10.06 | kernel |

## Where torch.compile wins

- One-sample norm, residual and the loss (N of a few thousand tokens) on sm86 / sm120:
  compile is faster, and the choice binds compile there (sm70: compile also beats
  TileLang above; its binding is pending the re-measure).
  - These rows are host bound: on the 5090 the Triton `rms_norm_mod` forward + backward
    costs about 430 us of host time against about 170 us of device time, compile about
    350 us.
  - The Triton path skips the custom-op dispatch (`_common.eager_function`), binds its
    tuned configs, caches the segment bounds per `seg` tensor and runs the
    parameter-gradient epilogue in one launch with the segment sums. What is left is the
    autograd engine, two Python `autograd.Function` nodes and two Triton launches per
    call, against Inductor's single generated graph.
  - Closing the gap needs fewer launches per block (fusing neighbouring ops) or a
    captured step, not a faster kernel.
- `token_mse`: launch-bound on every card; compile and Triton trade places.
- The TileLang row kernels reach 25-60 % of DRAM bandwidth on sm70 / sm75 with the
  per-element loads; the forward and dx-only backward also have a `vec` variant (one row
  staged through shared memory with tile copies) that the tuner times against them.
