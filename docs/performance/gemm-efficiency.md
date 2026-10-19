# GEMM efficiency per card

Achieved tensor TFLOP/s of the GEMM and flash-attention kernels of one training step,
held against two references:

- the clock peak: SMs x FLOP/clk/SM x the SM clock measured during that kernel;
- the measured ceiling: a large cuBLAS GEMM on the same card.

The cards are the RTX 5090, RTX 3090 and RTX 2080 Ti, plus the ceiling of the V100.

## Method

**Run:** the LoKr setup in fp16 (factor 16, full matrix), step 6, on the v0.1.0 tree.
The memory profile is `auto` on the RTX 3090 and 2080 Ti, and `ckpt-resident` on the
RTX 5090.

**Per kernel** (`kohakumoss_bench/measure/tflops.py`):

- **Profiling.** ncu application replay of the launches whose demangled name matches
  `gemm|flash`, inside the run's `PROFILE_STEPS` window, with
  `--clock-control none --cache-control none`.
- **FLOP.** HMMA warp instructions (`sm__inst_executed_pipe_tensor_op_hmma.sum`; the
  `..._subpipe_hmma_op_hmma` name on sm120) x the FLOP of one instruction, read from the
  kernel's mma shape: `16816` is 4096 and `1688` is 2048. Any other name takes the
  arch's fp16 shape (2048 on sm75; 4096 on sm86, sm89 and sm120).
- **Cross-check.** The fp16 -> fp32 tensor-op counter divided by the instruction FLOPs
  is 1.00 on every launch of all three cards.
- **Peak.** SMs x FLOP/clk/SM (table below) x the launch's own SM clock
  (`sm__cycles_elapsed.avg.per_second`).
- **Aggregation.** Per kernel name, TFLOP/s is weighted by FLOP; the clock and the peak
  are weighted by time.

**Ceiling** (`kohakumoss_bench/measure/ceiling.py`):

- `torch.mm` 8192^3 in fp16 and bf16, PyTorch's default fp32 accumulation, with
  reduced-precision reduction both allowed and disallowed.
- 3 s warm, then the median of 20 timed launches.
- The SM clock, power and temperature come from 20 ms nvidia-smi samples over the last
  1.5 s of the warm loop and the timed launches.
- The launched kernel is recorded by torch.profiler.

```
python -m kohakumoss_bench.measure.tflops run --config <train config with PROFILE_STEPS> \
    --out <dir> --sm 86 --sms 82 --ncu "sudo -n /usr/local/cuda/bin/ncu"
python -m kohakumoss_bench.measure.tflops report --report <dir>/tf.ncu-rep --sm 86 --sms 82
python -m kohakumoss_bench.measure.ceiling --out ceiling.json
```

`--inst-metric`, `--ops-metric`, `--flop-per-inst` and `--flop-per-clk` set the values
for an arch outside the tables. The output files are:

- `tflops.json`: every launch, the per-kernel rows, the totals and the cross-check ratio;
- `tflops.md`: the table.

## Peak per clock

Dense fp16-input / fp32-accumulate FLOP per clock per SM (`FLOP_PER_CLK`):

| arch | card | FLOP/clk/SM | datasheet figure it reproduces | measured ceiling |
| --- | --- | --- | --- | --- |
| sm70 | Tesla V100 | 1024 | 125 TFLOP/s = 80 SMs at 1.53 GHz | 937 (91.5 %) |
| sm75 | RTX 2080 Ti | 512 | 53.8 TFLOP/s = 68 SMs at 1.545 GHz | 490 (95.7 %) |
| sm86 | RTX 3090 | 512 | 71 TFLOP/s = 82 SMs at 1.695 GHz | 502 (98.0 %) |
| sm89 | RTX 4090 | 512 | 165.2 TFLOP/s = 128 SMs at 2.52 GHz | not measured |
| sm120 | RTX 5090 | 512 | 209.5 TFLOP/s = 170 SMs at 2.407 GHz | 487-489 (95.1-95.5 %) |

The GeForce cards run fp32 accumulation at half the fp16-accumulation rate. A datasheet
figure is this peak at the boost clock. Under load the clock differs from the boost
clock, so every row below uses the clock measured during that kernel.

## Measured ceilings

| card | dtype | TFLOP/s | SM clock (MHz) | power (W) / cap | FLOP/clk/SM | kernel |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | fp16 | 223.0 | 2685 | 575 / 575 | 489 | `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_nn` |
| RTX 5090 | bf16 | 230.5 | 2782 | 575 / 575 | 487 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_nn` |
| RTX 3090 | fp16 | 71.6 | 1740 | 369 / 370 | 502 | `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn` |
| RTX 3090 | bf16 | 72.2 | 1755 | 368 / 370 | 501 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` |
| RTX 2080 Ti | fp16 | 52.5 | 1575 | 250 / 250 | 490 | `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn` |
| RTX 2080 Ti | bf16 | 6.9 | 1500 | 249 / 250 | 68 | `magma_sgemmEx_kernel` (no bf16 tensor path on sm75) |
| Tesla V100 SXM2 16 GB | fp16 | 97.3 | 1297 | 297 / 300 | 937 | `cutlass_70_tensorop_f16_s884gemm_relu_f16_256x128_nn` |
| Tesla V100 SXM2 16 GB | bf16 | 10.6 | 1440 | 298 / 300 | 92 | `magma_sgemmEx_kernel` (no bf16 tensor path on sm70) |

- Every sustained GEMM runs at the card's power cap, and the cap sets the clock.
- Allowing or disallowing reduced-precision reduction moves no row by more than 3 %.
- Every 16-bit kernel accumulates in fp32 (the `s` in the cuBLAS / CUTLASS name).

## Per kernel

Columns:

| column | meaning |
| --- | --- |
| share | share of the step's GEMM + flash time |
| clock | the kernel's time-weighted SM clock, GHz |
| % peak | TFLOP/s against the clock peak at that clock |
| % datasheet | TFLOP/s against the datasheet figure |
| % ceiling | TFLOP/s against the fp16 ceiling above |

### RTX 5090 (170 SMs)

| kernel | share % | TFLOP/s | clock | % peak | % datasheet | % ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_tn` | 26.5 | 234.6 | 2.865 | 94.1 | 112 | 105 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x256_32x4_tn` | 20.6 | 242.4 | 2.908 | 95.8 | 116 | 109 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_nn` | 18.0 | 239.0 | 2.881 | 95.3 | 114 | 107 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x256_32x4_nn` | 5.2 | 238.9 | 2.911 | 94.3 | 114 | 107 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x128_64x3_nt` | 1.5 | 203.0 | 2.689 | 86.7 | 97 | 91 |
| `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn` | 1.4 | 85.6 | 2.690 | 36.6 | 41 | 38 |
| `flash_fwd_kernel` | 10.4 | 230.6 | 3.112 | 85.1 | 110 | 103 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | 12.4 | 238.4 | 3.049 | 89.8 | 114 | 107 |
| all GEMMs | 76.6 | 231.0 | 2.868 | 92.5 | 110 | 104 |
| all GEMM + flash | 100 | 230.4 | 2.916 | 90.8 | 110 | 103 |

### RTX 3090 (82 SMs)

| kernel | share % | TFLOP/s | clock | % peak | % datasheet | % ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn` | 25.1 | 69.1 | 1.751 | 94.0 | 97 | 97 |
| `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_tn` | 19.4 | 71.2 | 1.751 | 96.9 | 100 | 99 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn` | 14.5 | 71.8 | 1.751 | 97.7 | 101 | 100 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_nn` | 7.2 | 71.0 | 1.751 | 96.6 | 100 | 99 |
| `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_nt` | 5.5 | 26.2 | 1.754 | 35.6 | 37 | 37 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn` | 2.0 | 65.3 | 1.750 | 88.9 | 92 | 91 |
| `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_32x1_tn` | 1.4 | 22.0 | 1.744 | 30.0 | 31 | 31 |
| `flash_fwd_kernel` | 10.1 | 65.4 | 1.751 | 89.0 | 92 | 91 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | 12.3 | 67.1 | 1.751 | 91.2 | 95 | 94 |
| all GEMMs | 77.3 | 66.0 | 1.751 | 89.7 | 93 | 92 |
| all GEMM + flash | 100 | 65.8 | 1.751 | 89.5 | 93 | 92 |

### RTX 2080 Ti (68 SMs)

The sm75 attention is a TileLang kernel, which the `gemm|flash` filter does not select.

| kernel | share % | TFLOP/s | clock | % peak | % datasheet | % ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 40.1 | 59.1 | 1.756 | 96.6 | 110 | 113 |
| `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_tn` | 20.8 | 56.8 | 1.741 | 93.7 | 106 | 108 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn` | 16.7 | 58.3 | 1.756 | 95.4 | 108 | 111 |
| `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_nn` | 13.6 | 58.2 | 1.734 | 96.4 | 108 | 111 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nt` | 3.0 | 43.5 | 1.754 | 71.2 | 81 | 83 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_tn` | 2.4 | 47.1 | 1.755 | 77.1 | 88 | 90 |
| `cutlass_75_tensorop_f16_s1688gemm_relu_f16_64x64_tn` | 0.6 | 20.5 | 1.756 | 33.5 | 38 | 39 |
| all GEMMs | 100 | 56.9 | 1.749 | 93.4 | 106 | 108 |

## Reading

- **The large GEMMs are at the clock peak.** On all three cards they run at 93.7-97.7 %
  of the clock peak, and the flash kernels at 85-91 %. On the RTX 3090 the four large
  GEMMs (66 % of the GEMM + flash time) run at 97-100 % of the measured ceiling, at the
  same 1.74-1.75 GHz clock as the ceiling.
- **The power cap sets the training clock.** A sustained GEMM holds the card at its
  power cap: the RTX 5090 runs at 2.65-2.78 GHz and the 2080 Ti at 1.53-1.58 GHz. Under
  ncu replay each launch runs alone between idle gaps and clocks higher (5090: 2.87-3.11
  GHz; 2080 Ti: 1.75 GHz). That is why "% ceiling" and "% datasheet" exceed 100 there.
  "% peak" is the figure that does not depend on the clock.
- **Weak kernels:**
  - the `nt` 256x128 GEMM on the RTX 3090: 35.6 %, 5.5 % of the time;
  - the small `wmma` 32x32 GEMMs: 30-37 % on the RTX 3090 and 5090;
  - the `nt` / staged `tn` variants on the 2080 Ti: 71-77 %.
- **Reading the ncu tensor-pipe figure.** `sm__pipe_tensor_op_hmma_cycles_active` reads
  half of "% peak" on sm86 and sm75: kernels at 93.7-97.7 % of the clock peak read
  46.8-48.8 %. So a step of fp32-accumulate GEMMs tops out at about 50 % on those cards.
  On sm120 the metric reads the same as "% peak" (94.8-96.2 % on the large GEMMs).
  - The whole-step tensor active of the [bench ncu pass](bench-matrix.md#ncu)
    (duration-weighted over every kernel of the step) is 39.6 % on the RTX 3090 and
    36.7 % on the 2080 Ti. That is 79 % and 73 % of what fp32-accumulate GEMMs can
    reach.
  - The figure is 70.0 % on the RTX 5090.
  - On the 2080 Ti the TileLang attention kernel holds 32.5 % of the step's kernel time,
    at 30.3 % tensor active.
