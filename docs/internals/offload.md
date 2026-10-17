# Memory profiles and offload

`kohakumoss.training.memory` turns a profile name into a `MemoryPlan` once, and
`apply_to_backbone` / `apply_inference_plan` install the engines the plan names. The
training loop then calls one composed `BlockRunner`; nothing branches on the plan per
step.

## Profiles

| profile | checkpoint | activation offload | base weights | weight stream | optimiser state | fused step | gradient offload |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `speed` | none | no | compute dtype | off | GPU | no | no |
| `balanced` | block | no | compute dtype | off | GPU | no | no |
| `low` | block | yes | int8 / fp8 (`quant`) | off | 8-bit | yes | no |
| `minimal` | block | yes | int8 / fp8 | ring (2 blocks) | host, streamed to the GPU | yes | yes |
| `auto` | the fastest of the above whose estimate (probe-calibrated) fits the device | | | | | | |

`optim_state`: `gpu` (fp32 masters and moments resident), `stream` (held in pinned host
memory, streamed through the GPU per parameter / per block; the update runs on the
GPU), `8bit`, `cpu` (`CPUAdamW`, update on the host).

`quant` resolves to fp8 on archs with fp8 GEMMs (sm89+) and int8 elsewhere. Every field
of `MemoryPlan` can be overridden (`MEMORY_OVERRIDES`); `resolve` lists every invalid
combination in one error. For full finetune (`trainable_base`) the base stays in the
compute dtype and `ring` becomes `swap`. For inference (`inference=True`) activation knobs
are dropped and streaming needs no checkpointing.

## Block runners

| runner | what it does |
| --- | --- |
| `PlainRunner` | `block(x, ctx)` |
| `CheckpointRunner("block")` | non-reentrant checkpoint of the whole block |
| `CheckpointRunner("sub-block")` | checkpoints each callable of `block.sub_blocks()` (attention half, FFN half) |
| `CheckpointRunner("selective")` | keeps matmul / attention outputs, recomputes the elementwise ops (`register_saved_op` adds custom kernels to the kept set) |
| `partial:<f>` | checkpoints the first `round(depth * f)` blocks only |
| `OffloadRunner` | saved tensors of the block go to pinned host memory |
| `StreamRunner` | the block's weights are streamed (`WeightStream`) |
| `StageRunner` | global block index for a pipeline stage |

## Activation offload

`ActivationOffload` installs `saved_tensors_hooks` per block. A saved tensor above
`min_bytes` is copied device-to-host on a side stream during forward (its storage stays
reserved for that stream until the copy finishes). In backward the first unpack of block
`i` fetches all of block `i`'s packets and prefetches block `i-1` on a second stream.
Parameters, buffers and streamed weight slots are excluded by storage pointer. Combined
with block checkpointing, the checkpoint's saved block inputs are the offloaded tensors.

## Weight streaming

`WeightStream` keeps every streamed `MossLinear` weight in (pinned) host memory and a ring
of `resident` device slots. `enter(i)` makes block `i` resident and prefetches `i+1`;
backward makes `i` resident again and prefetches `i-1`. A slot is overwritten only after
the compute stream records that it is done with the slot's previous block. Streaming
requires every streamed block to be checkpointed during training: a non-checkpointed
block would keep a slot tensor alive in its autograd graph across blocks.

- **Frozen bases (`ring`)**: host -> device only; quantized bases stream `q` and `scale`
  and dequantize on use.
- **Trainable blocks (`swap`)**: fp32 masters on the host are the parameters. Before
  the last micro-batch, each weight gradient goes device -> pinned host and is summed in
  fp32 on a host worker thread (`HostAccumulator`), off the autograd thread. On the last
  micro-batch `StreamedAdamW` steps each block inside backward: when block `i`'s
  backward starts, its state (master, moments, host gradient sum; with
  `optim_state="gpu"` only the gradient sum, the states stay resident) is prefetched
  into a 3-slot device ring; when its last weight gradient arrives AdamW runs on a side
  stream and the new master / moments / compute-dtype weights go back to the host. The
  next forward's H2D of block `i` waits on that write-back. With `optim_state="cpu"`
  the sums land in the masters' `.grad` (`flush_grads()`), `CPUAdamW` steps, and
  `refresh()` updates the streaming copies.
- **Pinned vs staged**: `pinned=False` copies through one pinned staging buffer per
  weight (pageable host memory, synchronous staging).

## Optimisers and fused step

The optimiser step runs on the GPU by default, including under every offload knob;
`CPUAdamW` is the opt-in host update.

- `MasterAdamW`: fp32 master / exp_avg / exp_avg_sq resident on the device,
  `torch._fused_adamw_` per parameter, the parameter gets the master cast to its dtype.
- `HostStateAdamW`: the same update with master and moments in pinned host memory. Each
  parameter's state goes through a 3-slot device ring: H2D, update and D2H each on a
  side stream. A slot is reused only after its previous write-back, a parameter's state
  is loaded only after its own last write-back, and the next parameter's state is
  prefetched in the order its updates came in the previous step (first step: reverse
  registration order). `finish()` makes the compute stream wait for the updates.
- `StreamedAdamW`: per-block GPU AdamW for `swap` blocks (see Weight streaming).
- `CPUAdamW` is a `torch.optim.Optimizer` whose fp32 masters and moments live on the host;
  gradients go device -> pinned host, the update uses `torch._fused_adamw_` on the CPU, and
  updated weights go back host -> device. `step_param(p)` updates one parameter.
- All four take torch-style parameter groups; each group's lr / betas / eps /
  weight_decay apply to its parameters, and `param_groups` holds the live dicts a
  schedule edits. They apply `decay_to_one` groups to their fp32 masters themselves
  (`handles_decay_to_one`), so the bundle's post-step decay is skipped for them.
- `FusedStep` registers post-accumulate-grad hooks that step each parameter as soon as its
  gradient is complete and free it, so only one block's gradients are alive.
  `PerParamOptimizer` gives any torch optimiser that per-parameter interface behind a
  torch-style `param_groups` view for schedules (one view dict per source group).

### Fused step and non-finite gradients

`FusedStepBundle.step_param(p)` runs one transform per gradient (unscale, a device-side
finite flag, per-parameter clip) and passes the flag to the update as `found_inf`:

- the offload AdamWs (`MasterAdamW`, `HostStateAdamW`, `StreamedAdamW`) hand it to
  `torch._fused_adamw_`, which leaves the master, both moments and the step count
  untouched when it is set; `CPUAdamW` skips the update;
- a per-parameter torch optimiser built with `fused=True` gets it as `found_inf`; any
  other per-parameter optimiser sees a zeroed gradient (momentum and weight decay still
  move it).

A parameter whose gradient is non-finite is therefore not updated; every other parameter
of that step was already updated in backward and stays updated. The scaler backs off and
the step is reported non-finite. There is no global gradient norm in this mode.

`step()` first runs the bundle's finishers: the compute stream waits for every side
stream that updated parameters inside backward (`GradOffload.finish`,
`HostStateAdamW.finish`, `StreamedAdamW.finish`), before the finite flag is read and
before the next forward reads the parameters.

### Wiring (`training/memory/optimizers.py`)

`build_memory_optim` is the one place that turns a plan into optimisers, for the trainer
(`build_optim`) and for `MemorySession` alike:

| plan | device parameters | `swap` block masters | step |
| --- | --- | --- | --- |
| `optim_state='gpu'` | `adamw` (torch AdamW; `master_adamw` in MemorySession) | `StreamedAdamW`, resident states | in backward for `swap`, `fused_step` or `grad_offload`; else at `step()` |
| `optim_state='stream'` | `HostStateAdamW` | `StreamedAdamW`, streamed states | same |
| `optim_state='8bit'` | bitsandbytes AdamW8bit | (`swap` rejected) | same |
| `optim_state='cpu'` | `CPUAdamW` | `CPUAdamW` after `flush_grads` | at `step()`; with `fused_step`, device parameters in backward |

`resolve` sets `fused_step` for `swap` with a GPU optim_state (blocks can only step inside
backward) and for `grad_offload` with a GPU optim_state (otherwise every summed gradient
comes back to the device before the step, and the peak holds them all), unless
`fused_step` is overridden; `swap` with a contradicting override is rejected.
`grad_offload` + `fused_step` + `CPUAdamW` finishes each gradient on the device and the
host update reads it from there (one extra round trip per parameter); without
`fused_step` the host update reads the host sums directly.

## Gradient offload with accumulation

`GradOffload` registers a post-accumulate-grad hook per device parameter: as soon as
autograd has the micro-batch's gradient it is copied device -> pinned host on a side
stream and the device gradient is freed; a host worker thread (`HostAccumulator`) keeps
an fp32 sum across the `GRAD_ACC` micro-batches. No full-size gradient buffer stays on
the device. On the last micro-batch (`arm(last=True, final=opt.step_param)`) each
parameter's host sum comes back through a 3-buffer device ring on an H2D stream, is
added to the fresh device gradient on an update stream, and the GPU optimiser steps
that parameter right there; `finish()` makes the compute stream wait for the updates. A
host optimiser reads the sums directly instead (`CPUAdamW.use_host_grads`). Streamed
`swap` blocks accumulate on the host through `WeightStream`.

## MemorySession and parameter groups

`MemorySession(model, plan, device, dtype, grad_acc, trainable_base, ...,
param_groups)` applies the plan (`apply_to_backbone`) and builds its bundle through the
trainer's `build_memory_optim`, for benches and tests on models outside the trainer
(the synthetic MOSS-shaped stack). The driver calls `backward(loss, micro)` per
micro-batch and `step()` once per optimiser step; `masters(model)` returns every fp32
master by name.

`param_groups` are torch-style groups over the model's parameters as given before the
session is built (parameters the plan replaces, such as streamed block weights, are
followed by name). Each group is split into its device part and its streamed part, and
both parts keep the group's hyperparameters and extra keys, so per-group rules (no
weight decay on biases / norms / 1-D parameters, decay-to-one, LoRA+ ratios) reach
every optimiser path. Every trainable parameter must be in a group.

## Combination matrix

Two matrices, both with a negative case per axis:

- `tests/memory/matrix/test_trainer_matrix.py` runs the real trainer (`entry.run`, tiny
  MOSS model, synthetic data, 2 steps) over adapter / full FT x stream (`off`; `ring`
  for adapters, `swap` for full FT) x activation offload x gradient offload x fused step
  x optimiser state (`gpu`, `stream`, `cpu`) x GRAD_ACC (1, 2), and compares every
  trainable tensor with the plain run (tolerance 1e-4 of the update, or 20x the plain
  run's own run-to-run spread on GPUs with non-deterministic kernels). Negatives: a 25%
  larger LR, the trainable set built before the plan (`swap` blocks untrained), fused
  CPUAdamW reading host lanes. Combinations the knobs reject (`swap` with a
  contradicting `fused_step`) must be rejected by `validate`.
- `tests/memory/matrix/test_session_matrix.py` trains the synthetic MOSS-shaped stack
  through `MemorySession` under checkpoint (5) x activation offload x stream (3) x
  gradient offload x GRAD_ACC (1, 4) x optimiser state x base precision (incl. int8 /
  fp8), against a plain `MasterAdamW` reference; invalid combinations must be rejected
  by `resolve`.

`tests/memory/offload/test_param_groups.py` checks every optimiser, alone and in the
session, against `torch.optim.AdamW` with four groups of different lr / weight decay /
betas / eps.

| device | bases | valid combinations (pass) | rejected |
| --- | --- | --- | --- |
| RTX 4090, Windows | fp16, int8, bf16, fp8 | 912 | 528 |
| CPU | fp32, int8 | 456 | 264 |
| V100, Linux | fp16, int8 | 456 | 264 |

`tests/parallel/test_param_groups.py`: 75 cases on the 4090 and on CPU (CUDA cases
skipped there), all passing on the V100 as well.

Rejections: weight streaming needs every block checkpointed (`none`, `partial`), and
a quantized base cannot be trained (full FT).

## Hot paths

Per-block, per-tensor and per-parameter paths are specialised at build time: runners
`bind(layers)` to a tuple of per-block callables, the offload / stream / gradient
engines bind their device-or-host copy and wait functions at construction, `arm()`
swaps the hook body once per micro-batch, and `QuantWeight.bind` sets `linear` to the
kernel or the dequantizing path. Per-step and build-time choices stay plain `if`s.

## Quantized base

`quantize_model` replaces accepted `MossLinear` bases with `QuantWeight` (per-output-row
scales: int8 `round(w / s)`, `s = max|w_row| / 127`; fp8 e4m3 `w / s`, `s = max|w_row| / 448`).
`QuantWeight.bind` resolves a `quant_linear` kernel for the device once; without one the
weight is dequantized per call. `split_quantized` / `attach_quantized` load ComfyUI fp8 /
int8 checkpoints (`scale_weight` or `weight_scale` keys). `quantize_torch_linears` does the
same for plain `nn.Linear` modules of a frozen text encoder.

## Text encoder on small cards

`apply_to_text_encoder` quantizes and/or layer-streams the encoder: `LayerStreamer` swaps
each decoder layer's parameters and buffers into a device slot in a forward pre-hook
(prefetching the next layer) and back to host in the forward hook. Layers of different
structure (linear- and full-attention layers) get separate slot pools.

## Memory model

`estimate(Workload, MemoryPlan)` adds weights, trainable state, saved activations, the
recompute window, text-refiner activations, dequant transients and IO, times a 1.08
fragmentation factor. Saved activations per token per block are `12·D + 3·F` elements
without checkpointing and `D` (the block input) with block checkpointing; `act_factor`
scales the within-block part (1.0 for the fused kernel ops, 3.6 measured for the eager
ops). `auto_plan` walks profiles fastest-first; a probe step's measured peak rescales
later estimates.

Measured on a V100 32 GB (PCIe gen3 x8), MOSS-Image base, random weights, fp16, one
4096 + 300-token sample, norm + embedder weights trainable, TileLang block ops:

| profile | ms / step | tokens / s | step peak GiB | resident GiB | estimate GiB |
| --- | --- | --- | --- | --- | --- |
| speed | 2513 | 1751 | 26.31 | 9.64 | 34.00 |
| balanced | 3392 | 1297 | 11.36 | 9.63 | 12.87 |
| low | 3667 | 1200 | 6.13 | 4.76 | 7.09 |
| minimal | 5446 | 808 | 3.61 | 0.53 | 2.54 |

`int8` runs dequantize per call on this card (no `quant_linear` kernel was registered);
`minimal` is bound by host-device copies (every block's int8 weights cross PCIe twice
per step, forward and backward, plus the activation packets).

Copy bandwidth and streaming cost on the same card (8 MOSS-sized synthetic blocks,
4400 tokens, frozen blocks):

| | pinned H2D | pinned D2H | pageable H2D | pageable D2H |
| --- | --- | --- | --- | --- |
| GB/s (256 MB) | 6.24 | 6.15 | 5.23 | 3.34 |

| case | ms / step | peak GiB | H2D GB / step |
| --- | --- | --- | --- |
| resident | 245 | 7.59 | — |
| block checkpoint | 355 | 3.44 | — |
| checkpoint + activation offload | 369 | 3.79 | — |
| checkpoint + stream (pinned) | 439 | 4.36 | 2.33 |
| checkpoint + stream (staged) | 676 | 4.71 | 2.33 |

Each VRAM knob in isolation (`scripts/bench/offload/knobs.py`, one process per case,
8 MOSS-sized synthetic blocks = 0.75 B parameters, 4400 tokens, GRAD_ACC = 2):

| knob | V100 32 GB fp16: ms / step | peak GiB | H2D / D2H GB per step | RTX 4090 Windows (4 blocks, bf16): ms / step | peak GiB |
| --- | --- | --- | --- | --- | --- |
| full FT, device AdamW | 832 | 6.92 | 0 / 0 | 301 | 4.01 |
| full FT, CPUAdamW (grads on device) | 1607 | 4.11 | 0 / 0 | 421 | 2.60 |
| full FT, gradient offload + CPUAdamW | 1808 | 2.71 | 0 / 3.02 | 455 | 1.90 |
| frozen blocks, checkpoint none | 490 | 7.59 | — | 187 | 3.94 |
| frozen blocks, checkpoint block | 708 | 2.56 | — | 317 | 1.76 |
| frozen blocks, checkpoint sub-block | 703 | 2.49 | — | 233 | 1.58 |
| frozen blocks, checkpoint selective | 542 | 4.04 | — | 204 | 2.39 |
| frozen blocks, checkpoint partial:0.5 | 601 | 4.75 | — | 211 | 2.52 |
| full FT, stream (swap) + gradient offload + checkpoint, 4400 tokens | 2363 (537 / 1k tok) | 1.65 | 6.54 / 4.03 | 513 (117 / 1k tok) | 1.55 |
| same, 17600 tokens | 4229 (240 / 1k tok) | 5.43 | 6.54 / 4.03 | 1407 (80 / 1k tok) | 5.03 |

Streaming moves the same bytes per step regardless of the token budget, so the larger
batch halves (V100) or one-thirds (4090) the time per token.

Under Windows WDDM on the RTX 4090 (4 blocks): pinned H2D 19.7 GB/s, pinned D2H
24.3 GB/s, pageable H2D 8.3 GB/s, pageable D2H 13.5 GB/s; checkpointed streaming costs
122 ms / step pinned vs 152 ms staged.

### Hardware-limit check

`scripts/bench/offload/limits.py` runs each combination in its own process (MOSS-sized
synthetic blocks, 4400 tokens, block checkpointing, `stream_resident=2`) and divides
the step time by `max(copy bound, compute)`: copy bound = the largest of H2D bytes /
pinned H2D bandwidth, D2H bytes / pinned D2H bandwidth and all bytes / measured
concurrent (both directions) bandwidth; compute = the resident step at the same
GRAD_ACC. 1.00 = the step costs only its bound.

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

- Streamed optimiser states move 12 bytes per parameter each way per step; the V100
  rows at 1.34-1.39 are copy-bound (profiled: update kernels fully hidden, the copy
  engines busy for the whole step).
- CPUAdamW is bound by the host update, not by PCIe or the GPU.
- `swap` with resident states and `ring` alone are copy-bound per phase: forward with a
  2-slot ring waits on each block's H2D, so the per-step sum understates the bound.
- `ring` + gradient offload on the V100 is slower than `ring` alone although only
  adapter-sized gradients go to the host: the H2D of the weight ring runs about 18%
  slower while D2H gradient copies are in flight.
- 4090 `swap` with resident states at 0.74: the resident reference runs its AdamW
  after backward over full-size gradients, while `swap` steps each block inside
  backward; the reference is therefore not a lower bound for in-backward steps.

## Windows and XPU

The engines take their streams and events from the device's module (`offload/accel.py`:
`torch.cuda` or `torch.xpu`), so CUDA and Intel XPU run the same side-stream paths;
other devices run the synchronous host paths.

| feature | Windows (CUDA) | XPU (Arc B50, Windows) |
| --- | --- | --- |
| memory profiles, estimate, `auto` planner | works | works |
| block runners (all checkpoint modes) | works | works |
| activation offload | works | works |
| weight streaming, pinned and staged | works | works |
| gradient offload + accumulation | works | works |
| GPU AdamW (resident, streamed states, streamed blocks), CPUAdamW, fused per-parameter step | works | works |
| `MemorySession`, parameter groups | works | works |
| quantized base (int8 / fp8), streamed quantized base | works (fp8 on sm89) | int8 |
| text-encoder layer streaming | works | works |
| inference plans (`apply_inference_plan`) | works | works |
| multi-GPU (DDP, FSDP2, pipeline, threaded) | Linux-only by design (rejected by knob validation) | same |

The XPU column is `tests/memory/offload` on the B50 (activation offload, streaming,
gradient offload, every optimiser alone and in the session).
