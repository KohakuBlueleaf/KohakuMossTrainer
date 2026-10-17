# Tuning

Kernels take their tile config from `kohakumoss_kernels.tuning.select(kernel, device,
key, candidates, launch)`:

1. **Planner.** `candidates()` enumerates the legal configs (shared memory, register
   and warp-partition limits of the card) and ranks them with `tuning.plan.rank`: a
   roofline estimate (MMA flops and exponentials vs the measured MMA rate, DRAM bytes
   vs the measured bandwidth) scaled by wave quantisation over the SMs. The attention
   kernels put the arch's measured seed tiles ahead of the ranking
   ([kernel-dev.md](kernel-dev.md#measured-design-per-arch)).
2. **Shortlist timing.** The top `shortlist` configs are timed with device events
   around plain launches (`timing="events"`): a busy warmup to boost clocks, then the
   best of ten calls, host launch cost included as the eager training loop pays it. The
   tuner never captures a CUDA graph: a capture in the training process makes CUDA
   calls of other threads (the DataLoader's pin-memory thread) fail, in global and, as
   measured on Windows, thread-local capture mode alike. `timing="graph"` (CUDA-graph
   replay, launch cost excluded) is for benches. A config that raises a build / launch
   error (`errors.BUILD_ERRORS`) is recorded and skipped; out-of-memory always
   propagates; when every config fails, `ConfigsFailed` hands the op to the registry
   fallback (docs/internals/kernel-dev.md). Shortlist per kernel:

   | kernel | shortlist |
   | --- | --- |
   | Triton attention (fwd, dK/dV, dQ, fused bwd) | 10 |
   | TileLang attention (fwd, dK/dV, dQ; each config is an nvcc compile) | 6 |
   | Triton attention backward strategy (fused vs split) | both |
   | attention race (Triton, TileLang, torch `varlen_attn`) | every implementation |
   | fused linear, TileLang block ops | every legal config |
   | quant matmul (Triton, TileLang) | every tile plus the PyTorch paths |
   | kernel-vs-compile race of a block op | both |
   | anything else (`select` default) | 4 |
3. **Table.** The winner, every timing and every failure are written to
   `<cache>/tuning/<card-sm-cuda-torch>/<tag>/<kernel>.json` (atomic replace), where
   `<tag>` hashes the kohakumoss / triton / tilelang versions. Later processes read the
   table and never time again; a stored config that is no longer among the current
   candidates is re-tuned.

Problem keys bucket the variable dimensions, so a training run with varying resolutions
settles into a handful of keys. Attention keys are `<dtype>-h<heads>-d<head dim>-L<longest
segment rounded up to a power of two, at least 64>`, plus `-split` for the Triton
kernels in the `split` precision mode; the token and segment counts are
not part of the key (the grid covers them, the tiles do not depend on them).

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

`tuning.broadcast(src=0)` replaces every rank's tables with rank `src`'s (memory and
disk) over the initialised process group, then runs `tuner.reset()`: settled choices and
every binding cache registered with `tuner.on_reset` (raced block ops, fused-linear
runners, attention and quant bindings) are dropped, so live calls bind from the shared
tables. The trainer prewarms on rank 0 only, broadcasts, then starts; with identical
cards every rank runs the same configs (a rank on a different card has no rows for its
card and tunes its own).

`KOHAKUMOSS_KERNEL_CACHE` moves the cache root (default `~/.cache/kohakumoss_kernels`).
