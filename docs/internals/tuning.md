# Tuning

Kernels take their tile config from `kohakumoss_kernels.tuning.select(kernel, device,
key, candidates, launch)`:

1. **Planner.** `candidates()` enumerates the legal configs (shared memory, register
   and warp-partition limits of the card) and ranks them with `tuning.plan.rank`: a
   roofline estimate (MMA flops and exponentials vs the measured MMA rate, DRAM bytes
   vs the measured bandwidth) scaled by wave quantisation over the SMs.
2. **Shortlist timing.** The top few configs (4 for Triton, 3 for TileLang, whose
   configs are each an nvcc compile) are timed with `tuning.timing.graph_ms`: a busy
   warmup to boost clocks, then CUDA-graph replay so host launch cost is excluded. A
   config that fails to build or launch is recorded and skipped.
3. **Table.** The winner, every timing and every failure are written to
   `<cache>/tuning/<card-sm-cuda-torch>/<kernel>.json` (atomic replace). Later
   processes read the table and never time again.

Problem keys bucket the variable dimensions (attention: dtype, heads, head dim, segment
count capped at 64, and the longest segment rounded up to a power of two), so a
training run with varying resolutions settles into a handful of keys.

## Device description

`hardware.device.probe(device)` returns the card's driver limits plus three measured
rates (16-bit GEMM TFLOP/s with fp32 accumulation, DRAM and L2 copy GB/s), measured
once per card / CUDA / torch combination and cached in `<cache>/devices/`. Inside a
trace or graph capture nothing is measured: planners use the unmeasured `query`
description with default rates.

## Modes

`KOHAKUMOSS_TUNE` or `tuning.set_mode(...)`:

| mode | behaviour |
| --- | --- |
| `auto` | a new key on an eager call times the shortlist and stores the winner |
| `off` | the planner's first candidate, never timed or stored |
| `prewarm` | as `auto`; the trainer drives it with representative shapes before step 0 |

Inside `torch.compile` tracing or CUDA-graph capture a missing key takes the first
candidate untimed (not stored), so a compiled graph carries no tuning code.

## Multi-process

`tuning.broadcast(src=0)` copies rank `src`'s tables to every rank (memory and disk)
over the initialised process group. The trainer prewarms on rank 0, broadcasts, then
starts; every rank then runs the same configs.

`KOHAKUMOSS_KERNEL_CACHE` moves the cache root (default `~/.cache/kohakumoss_kernels`).
