# Offload engines on synthetic blocks (V100, RTX 4090)

Step time, peak memory and copy volume of each offload knob, and how close each
combination comes to its hardware bound, on a V100 32 GB and an RTX 4090. The blocks are
MOSS-sized synthetic blocks: they measure the copy and compute cost of the engines, not
model output. The real-model numbers on an RTX 5090 are in
[offload.md](../internals/offload.md#real-model-benches-rtx-5090); the engines themselves
are described there too.

How to read the tables:

| term | meaning |
| --- | --- |
| ms / step | one optimiser step |
| peak GiB (alloc) | the torch allocator's allocated-bytes peak (live tensors, not reserved or device use) |
| H2D / D2H GB | host-to-device / device-to-host bytes per step |
| overlap | step time / `max(copy bound, compute)`; 1.00 = the step costs only its bound |
| - | no copy of that kind in the case |

Benches: `scripts/bench/offload/{stream,knobs,limits}.py`, one process per case.

## Copy bandwidth and streaming cost (V100)

MOSS-sized synthetic blocks (copy / compute cost only, no model output), tree a3cfbe0.
V100 32 GB (PCIe gen3 x8), 8 blocks, 4400 tokens, frozen blocks:

| | pinned H2D | pinned D2H | pageable H2D | pageable D2H |
| --- | --- | --- | --- | --- |
| GB/s (256 MB) | 6.24 | 6.15 | 5.23 | 3.34 |

| case | ms / step | peak GiB (alloc) | H2D GB / step |
| --- | --- | --- | --- |
| resident | 245 | 7.59 | - |
| block checkpoint | 355 | 3.44 | - |
| checkpoint + activation offload | 369 | 3.79 | - |
| checkpoint + stream (pinned) | 439 | 4.36 | 2.33 |
| checkpoint + stream (staged) | 676 | 4.71 | 2.33 |

## Each VRAM knob in isolation

MOSS-sized synthetic blocks (copy / compute cost only, no model output), tree a3cfbe0.
`scripts/bench/offload/knobs.py`, one process per case, 8 blocks (0.75 B parameters) on
the V100 in fp16 and 4 blocks on the RTX 4090 (Windows) in bf16, 4400 tokens,
GRAD_ACC 2:

| knob | V100 32 GB fp16: ms / step | peak GiB (alloc) | H2D / D2H GB per step | RTX 4090 Windows (4 blocks, bf16): ms / step | peak GiB (alloc) |
| --- | --- | --- | --- | --- | --- |
| full FT, device AdamW | 832 | 6.92 | 0 / 0 | 301 | 4.01 |
| full FT, CPUAdamW (grads on device) | 1607 | 4.11 | 0 / 0 | 421 | 2.60 |
| full FT, gradient offload + CPUAdamW | 1808 | 2.71 | 0 / 3.02 | 455 | 1.90 |
| frozen blocks, checkpoint none | 490 | 7.59 | - | 187 | 3.94 |
| frozen blocks, checkpoint block | 708 | 2.56 | - | 317 | 1.76 |
| frozen blocks, checkpoint sub-block | 703 | 2.49 | - | 233 | 1.58 |
| frozen blocks, checkpoint selective | 542 | 4.04 | - | 204 | 2.39 |
| frozen blocks, checkpoint partial:0.5 | 601 | 4.75 | - | 211 | 2.52 |
| full FT, stream (swap) + gradient offload + checkpoint, 4400 tokens | 2363 (537 / 1k tok) | 1.65 | 6.54 / 4.03 | 513 (117 / 1k tok) | 1.55 |
| same, 17600 tokens | 4229 (240 / 1k tok) | 5.43 | 6.54 / 4.03 | 1407 (80 / 1k tok) | 5.03 |

- Streaming moves the same bytes per step whatever the token budget, so the larger
  batch halves (V100) or cuts to a third (4090) the time per token.
- Under Windows WDDM on the RTX 4090 (4 blocks): pinned H2D 19.7 GB/s, pinned D2H
  24.3 GB/s, pageable H2D 8.3 GB/s, pageable D2H 13.5 GB/s. Checkpointed streaming
  costs 122 ms / step pinned against 152 ms staged.

## Hardware-limit check

MOSS-sized synthetic blocks (copy / compute cost only, no model output), tree a3cfbe0.
`scripts/bench/offload/limits.py`, one process per case, 4400 tokens, block
checkpointing, `stream_resident=2`. The copy bound is the largest of H2D bytes / pinned
H2D bandwidth, D2H bytes / pinned D2H bandwidth and all bytes / measured concurrent
(both directions) bandwidth; compute is the resident step at the same GRAD_ACC.

| case | V100 32 GB, 8 blocks, fp16: ms | overlap | RTX 4090 Windows, 4 blocks, bf16: ms | overlap |
| --- | --- | --- | --- | --- |
| full FT resident, GPU AdamW | 459 | 1.00 | 228 | 1.00 |
| same, GRAD_ACC 4 | 1756 | 1.00 | 564 | 1.00 |
| gradient offload, GPU AdamW in backward, GRAD_ACC 4 | 2116 | 1.20 | 584 | 1.03 |
| optimiser states streamed (`stream`), fused step | 2474 | 1.34 | 416 | 1.18 |
| CPUAdamW | 1220 | 2.66 | 260 | 1.14 |
| `swap`, GPU AdamW, resident states | 666 | 1.45 | 170 | 0.74 |
| `swap`, GPU AdamW, streamed states | 3068 | 1.35 | 457 | 1.07 |
| same, GRAD_ACC 4 | 5190 | 1.39 | 925 | 1.40 |
| `swap`, CPUAdamW | 1558 | 3.40 | 285 | 1.25 |
| adapter (frozen blocks) resident | 354 | 1.00 | 120 | 1.00 |
| adapter, activation offload | 355 | 1.00 | 120 | 1.00 |
| adapter, `ring` | 439 | 1.24 | 118 | 0.98 |
| adapter, `ring` + activation offload | 452 | 1.22 | 126 | 1.06 |
| adapter, `ring` + gradient offload | 501 | 1.42 | 116 | 0.97 |
| adapter, `ring` + activation + gradient offload | 512 | 1.38 | 125 | 1.05 |
| adapter, `ring`, int8 base | 389 | 1.10 | 131 | 1.10 |

Bandwidth: V100 pinned H2D 6.70 GB/s, D2H 6.61 GB/s, both directions 9.80 GB/s total;
RTX 4090 (WDDM) 25.1 / 26.3 GB/s, both directions 25.7 GB/s total (no duplex gain).
MMA peak 90 TFLOP/s (V100 fp16), 175 TFLOP/s (4090 bf16).

- **Streamed optimiser states** move 12 bytes per parameter each way per step. The V100
  rows at 1.34-1.39 are copy-bound: profiled, the update kernels are fully hidden and
  the copy engines are busy for the whole step.
- **CPUAdamW** is bound by the host update, not by PCIe or the GPU.
- **`swap` with resident states and `ring` alone** are copy-bound per phase: forward with
  a 2-slot ring waits on each block's H2D, so the per-step sum understates the bound.
- **`ring` + gradient offload on the V100** is slower than `ring` alone although only
  adapter-sized gradients go to the host: the H2D of the weight ring runs about 18%
  slower while D2H gradient copies are in flight.
- **4090 `swap` with resident states at 0.74:** the resident reference runs its AdamW
  after backward over full-size gradients, while `swap` steps each block inside
  backward. The resident reference is therefore not a lower bound for in-backward
  steps.
