# Multi-GPU training

Set `PARALLEL` in the train config; the launcher spawns one process per entry of
`DEVICES` (see [distributed launch](../internals/distributed-launch.md)).

Multi-GPU training is Linux-only by design: on Windows, knob validation rejects any
`PARALLEL` other than `"none"` and more than one entry in `DEVICES`.

| `PARALLEL` | engine | when |
| --- | --- | --- |
| `ddp` | `GradReducer` / `AsyncReducer`: bucketed SUM all-reduce of the trainable gradients, overlapped with backward, under every memory profile | LoRA / LyCORIS (only adapter gradients cross the wire); full finetune with weight streaming |
| `fsdp` | FSDP2 `fully_shard` per conditioning linear and per block (joint and text-refiner blocks), then the root, every parameter in its own dtype; adapters are sharded with the module they adapt | full finetune that does not fit one card |
| `pipeline` | `torch.distributed.pipelining` stages over contiguous block ranges | cards without a fast interconnect, or a model that does not fit one card |
| `threaded` | one process, stages on different devices | mixed GPUs (e.g. 3090 + 2080 Ti) |

Parallel layers: `configs/moss/parallel/parallel_{ddp,fsdp,pipeline,threaded}.py`.

## Run

```bash
# adapters over two GPUs (the ddp layer); --algo full gets fsdp where the sharded
# footprint fits the card, else ddp over the swap profile (full-finetune.md)
kohakumoss-config new train --weights models/MOSS-Image-Base --data <dataset> --gpus 2 --name my-lora
kogine run scripts/train/moss.py --config configs/user/my-lora.py

# a shipped recipe (set WEIGHTS / DATASETS in a file that use_config's it)
kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_ddp.py
```

- For `pipeline` or `threaded`, compose `configs/moss/parallel/parallel_pipeline.py` (or
  `parallel_threaded.py`) with `use_config` after the VRAM layer of a train config
  ([configs.md](configs.md)).
- `kogine run` is the launch: the train script spawns one process per device itself.
  `DIST_BACKEND="auto"` picks nccl on CUDA.
- `scripts/dist/check_comm.py` checks the process group and measures every link before a
  long run.

## Supported memory plans

Knob validation resolves the MemoryPlan and rejects every combination outside this
table with one message each:

| `PARALLEL` | stream | optim_state | base_precision | act_offload | fused_step / grad_offload |
| --- | --- | --- | --- | --- | --- |
| `none` | any | any | any | any | any |
| `ddp` | any (`swap`: one node, shared host masters) | any | any | any | any |
| `fsdp` | `off` | `gpu` | `compute` | no | no |
| `pipeline`, `threaded` | `off` | `gpu` | `compute` | no | no |

One line per mode, what it runs:

- `ddp`: every memory plan, offload included (stock torch DDP: every plan whose
  trainable gradients stay on the device until the step, so no streamed base, no
  `fused_step`, no `grad_offload`).
- `fsdp`: checkpointing (`CHECKPOINT` block / selective / partial:<f>), no memory offload.
- `pipeline`: checkpointing (every `CHECKPOINT`), no memory offload.
- `threaded`: checkpointing (every `CHECKPOINT`; the async schedule with per-micro-batch
  adapter state needs `gpipe` or `checkpoint='none'`), no memory offload.

Each refusal names what the mode runs instead.

- `ddp` runs every memory profile that runs on one GPU (`ckpt-ring*`, `*-gradoff`,
  `*optfused*`, `*-optcpu`, `*-optstream`, `low`, `minimal`, the full-finetune `swap`
  profiles); every trainable gradient is summed across ranks before the optimiser
  reads it ([Gradients that leave backward early](#gradients-that-leave-backward-early)).
- `fsdp` runs no sampling during training (`SAMPLE_EVERY=0`): the sample forward runs
  on rank 0 only and cannot all-gather the sharded blocks. Validation runs on every
  rank. `checkpoint='sub-block'` is rejected under `fsdp` (the block parts bypass the
  FSDP2 unshard on the block's `__call__`).
- FSDP2 checkpoints hold full tensors: every rank gathers the trainable / EMA tensors
  and the optimiser state (`get_optimizer_state_dict(full_state_dict=True)`), rank 0
  writes; resume re-shards them. The final export runs with every unit unsharded
  and reads the live parameters.

## Loss normalisation

The loss of every rank is divided by the global token count of the step, so gradients
are summed across ranks and the logged loss is the sum of the ranks' shares:

- `GradReducer` sums; `fsdp.shard` sets the FSDP2 divide factor to 1 and forces plain
  SUM collectives.
- The loss shares and the non-finite flags of the step go in one SUM all-reduce under
  DDP and FSDP2.
- Under the pipelines every stage sees the same step: the loss of each micro-batch is
  divided by the step's token count on the last stage, micro-batch losses are summed
  (the schedule runs with `scale_grads=False`), and the step's loss is broadcast from
  the rank holding the last stage (rank 0 under the V schedules).

Under FSDP2 with `GRAD_ACC > 1`, `FSDP_REDUCE` chooses when gradients are
reduce-scattered:

| `FSDP_REDUCE` | reduce-scatter | gradients held between micro-batches |
| --- | --- | --- |
| `"step"` (default) | on the last micro-batch only (`set_requires_gradient_sync`); one per step | the unsharded gradients of the whole model (4 B per parameter for fp32 masters; the memory estimate counts them) |
| `"micro"` | after every micro-batch; `GRAD_ACC` per step | sharded gradients only |

`FSDP_RESHARD` sets when a block's gathered parameters are freed: `"forward"` (default,
ZeRO-3) after its forward, gathered again in backward; `"backward"` (ZeRO-2) after its
backward, one gather per micro-batch with every block held unsharded at the start of
backward (the memory estimate counts them).

## Data parallel (`ddp`)

`GradReducer` groups the trainable parameters (reverse order, same dtype and device)
into buckets of 25 MiB:

- At the step's first micro-batch every bucket buffer is zeroed and becomes the
  gradients of its parameters (each `.grad` is a view of the buffer), so autograd
  accumulates in place in the bucket.
- On the last micro-batch the post-accumulate hook of a bucket's last parameter launches
  its async SUM all-reduce, which overlaps the rest of backward; `finish` (before the
  optimiser step) makes the compute stream wait for every bucket.
- A gradient that something replaced with its own tensor is copied back into the bucket
  before the launch.
- Buckets launch in bucket order (a complete bucket waits for the ones before it), so
  every rank issues the same collectives in the same order even when a gradient is
  missing on one rank.

### Gradients that leave backward early

When the memory plan moves or consumes a gradient before the optimiser step, the
`AsyncReducer` (`kohakumoss.parallel.ddp`) sums it first, on its own side stream, so
the reduction overlaps the rest of backward. On the last micro-batch:

| plan | where the gradient goes | reduction |
| --- | --- | --- |
| stream `off` / `ring`, no fused step, no grad offload | stays on the device until the step | `GradReducer` buckets (above) |
| `fused_step` | the per-parameter step inside backward | the post-accumulate hook copies the gradient into its static `AsyncReducer` bucket (reverse order, one dtype, 25 MiB); a complete bucket is all-reduced asynchronously and its parameters step on the side stream once it lands |
| `grad_offload` (+ `fused_step`) | host fp32 sums across micro-batches | the last micro-batch adds the host sum to the device gradient (as on one GPU), then the bucket reduction; the summed gradient then steps in backward (fused) or replaces the host sum (D2H on the side stream) for the host / device optimiser |
| `swap` + `optim_state` `stream` | the per-block `StreamedAdamW` step inside backward, on the block's owner | each block's fp32 gradient sums (host sum of earlier micro-batches + this one) are one flat buffer per ring slot, reduced onto the block's owner on the update stream; the owner alone transforms and updates it ([Shared host masters](#shared-host-masters)) |
| `swap` + `optim_state` `gpu` | the per-block `StreamedAdamW` step inside backward, every rank (resident states) | the same flat buffer, all-reduced; every rank updates its device states, the owner writes the shared host master and streaming copy |
| `swap` + `optim_state='cpu'` | host masters stepped by the host AdamW of the block's owner | each block's gradients (host sum + this micro-batch) are gathered into one fp32 flat buffer on a side stream, reduced onto the owner and copied to its host gradient |

Earlier micro-batches accumulate exactly as on one GPU (in place, or on the host); only
the last one's total crosses the wire.

### Reduction dtype

`DDP_REDUCE_DTYPE="fp32"` (default, the rule FSDP2 follows) sums in fp32:

- a 16-bit bucket is copied to fp32 at its launch and the sum copied back once it lands;
- a 16-bit gradient leaving backward early goes into an fp32 bucket and comes back in
  its parameter's dtype;
- the streamed blocks' flat buffers are fp32 already.

16-bit sums over several ranks would lose precision with the world size, and fp16
loss-scaled gradients could overflow in the sum where no rank's did. The fp32 copy of a
16-bit bucket lives from its launch to the end of the step's reductions (all 16-bit
trainable gradients at once at the end of backward; trainable adapters and fp32 masters
have none).

`DDP_REDUCE_DTYPE="param"` sums in each parameter's dtype. Every collective runs on the
device (NCCL).

### Step end

`finish` (before the step):

- launches the buckets still open. A gradient absent on this rank counts as zero; one
  absent on every rank is not delivered, so it is not stepped, as on one GPU: every
  bucket carries one presence count per parameter through its all-reduce, read on the
  host only for a bucket with a parameter missing locally;
- makes the compute stream wait for the side stream and the update streams. That wait
  is `ddp_comm_wait_ms` (it includes the in-backward updates queued behind the last
  reduction); `ddp_span_ms` runs from the last micro-batch's start to the end of that
  wait, and `ddp_reduce_mb` counts every byte reduced in the step.

Host-stepped gradients (`optim_state='cpu'`, gradient offload) are synchronised to the
host once at the step. The initial broadcast covers the trainable device parameters
(rank 0's values); the streamed masters are one shared copy already.

### Stock torch DDP

`DDP_BACKEND="torch"` runs the step under stock torch `DistributedDataParallel`
(`wrap_torch_ddp`) instead of the project's reducers, as an A / B baseline:

- the train step calls the wrapper, `no_sync()` covers every micro-batch but the last,
  and buffers are not broadcast per forward;
- a comm hook makes its bucket reduction a SUM (in `DDP_REDUCE_DTYPE`) like the
  project's;
- it reduces only gradients that stay on the device until the step: knob validation
  rejects it with `fused_step`, `grad_offload`, a streamed base, or
  `DROPOUT['module'] > 0` (an unused parameter stops torch DDP's reducer).

### Shared host masters

Under DDP with `stream='swap'` (one node, Linux) the ranks share one copy of the fp32
host masters and of their compute-dtype streaming copies:

- Rank 0 creates a `/dev/shm` region (`kohakumoss.offload.weights.shared.SharedHost`,
  page-locked with `cudaHostRegister`) and fills it from the loaded weights; the other
  ranks map it before the step loop, and the file is unlinked once every rank holds the
  mapping.
- Block `b` belongs to rank `b % world` (`block_owners`): its gradient is reduced (not
  all-reduced) onto that rank, which alone holds its host optimiser state and steps it
  (ZeRO-1 on the host), and writes the new master and streaming copy into the shared
  region.
- Every rank meets at a barrier after the step's write-backs, so the next forward reads
  the new weights from the shared region.
- The global gradient norm adds the owners' squared norms over the ranks (`ShardedNorm`,
  `sum_block_stats`); checkpoints keep each rank's optimiser shard in
  `optim.<rank>.pt`.

Host RAM of full-finetune streaming in bytes per streamed block parameter, W ranks on
one node (16-bit streaming copies, GRAD_ACC = 1):

| `optim_state` | one GPU | DDP per rank (shared region + own shard) | DDP whole node |
| --- | --- | --- | --- |
| `stream` (`StreamedAdamW`) | masters 4 + copy 2 + moments 8 = 14 | 6 shared + 8 / W | 14 |
| `cpu` (`CPUAdamW`) | 6 + optimiser 20 (master, moments, grad, shadow) + host grad 4 = 30 | 6 shared + 24 / W | 30 |
| `gpu` (resident states) | 6 | 6 shared | 6 |

With `GRAD_ACC > 1` every rank also sums its own micro-batches' block gradients on the
host before the last one's reduction (4 per rank, all blocks). The rank report logs
`host_rss_gib`, `host_private_gib` and `host_shared_gib` per rank and the job's
`host_ram_gib` (private parts summed, the shared part once).

## FSDP2 (`fsdp`)

`fsdp.shard(backbone, block_units(backbone), conditioning_units(backbone))` makes one
unit of each conditioning linear (`time_embedding.0`, `time_embedding.2`,
`shared_block_modulation`, `video_head_modulation`), every text-refiner block and every
joint block, then the root (embedders, head).

Dtypes:

- Every parameter is gathered in its own dtype, the dtype a single-GPU run computes
  with: the 16-bit frozen base, the fp32 KEEP_FP32 conditioning, the fp32 masters of
  adapters and full-finetune weights (16-bit under `OPTIMIZER='adamw16'`). Gradients are
  reduced in fp32.
- FSDP2 reduce-scatters one gradient dtype per unit, so the conditioning linears, which
  hold fp32 parameters next to the root's 16-bit ones under `adamw16`, are units of
  their own (an adapter lives in the unit of the module it adapts). For the same reason
  `PARALLEL='fsdp'` with `adamw16` accepts only those conditioning modules in
  `KEEP_FP32`.
- A unit holding parameters of several dtypes all-gathers them as bytes (FSDP2's uint8
  path). Gathering fp32 masters in fp32 costs twice the all-gather bytes of a 16-bit
  gather for full-finetune weights; adapter masters are small.
- Forward inputs keep their dtype (`cast_forward_inputs=False`: sigma stays fp32, an fp32
  residual stream stays fp32).

Overlap and reduction:

- The root and the conditioning units keep their parameters gathered between forward
  and backward; the root prefetches the conditioning units and the first block, and
  every block prefetches the next block's all-gather in forward and the previous
  block's in backward (`prefetch=1`).
- Gradients are a plain SUM (`set_gradient_divide_factor(1.0)` with
  `set_force_sum_reduction_for_comms(True)`), which is NCCL's zero-copy reduction and
  also what gloo implements.

## Pipeline (`pipeline`)

### Memory

Each rank builds the whole model on its device, keeps its stages' layers, and moves the
other layers (and top-level modules with no local parameter) to host memory.

- Checkpoints merge every rank's trainable / EMA entries on rank 0 and keep each rank's
  optimiser state. Every rank moves its tensors to the host before `gather_object`.
- The final export first merges every rank's trained stage entries into rank 0's live
  trainable set (`PipelineParts.before_export`, a collective), so the exported adapters
  / merged weights carry every stage's training.

### Schedules

`PIPELINE_SCHEDULE` (default `1f1b`):

| schedule | stages per rank | `MICROBATCHES` |
| --- | --- | --- |
| `gpipe`, `1f1b` | one | at least the rank count |
| `interleaved-1f1b`, `looped-bfs`, `interleaved-zb` | two (looped: rank `r` holds stages `r` and `r + W`) | at least the rank count; `interleaved-1f1b` / `interleaved-zb`: a multiple of the round count `max(1, MICROBATCHES // W)` |
| `zbv`, `dualpipe-v` | two (V: rank `r` holds stages `r` and `2W-1-r`) | at least the rank count (`dualpipe-v`: twice that) |

- The bubble of `gpipe` / `1f1b` is `(W-1) / (MICROBATCHES + W - 1)` of the step; `1f1b`
  holds at most `W` micro-batches of activations on the first stage where `gpipe` holds
  all of them.
- The zero-bubble schedules (`interleaved-zb`, `zbv`, `dualpipe-v`) run a stage's
  backward as an input-gradient pass and a later weight-gradient pass over one retained
  graph, so under them compiled backwards keep their buffers (`retained_backward`, set
  before the first compile).
- The retained graph also keeps every checkpointed block's recomputed activations from
  the input pass until the weight pass, so these schedules hold the full
  (uncheckpointed) activations of every micro-batch between its two passes: on 24 GB
  cards with MOSS LoRA, `zbv` runs out of memory at 4608- and 9216-token micro-batches
  where `1f1b` peaks at 11 GiB.

### What crosses a stage boundary

- Every rank draws the same step (the data, sigma and noise streams of rank 0) and
  builds every micro-batch's layout itself; the first step checks that the ranks agree
  (sizes and sigma bits, all-gathered once) and raises on every rank otherwise.
- Only `(x, modulation, head modulation)` cross a boundary. `x` travels in the
  residual-stream dtype (fp32 under the fp32-residual fix); the modulations carry a
  gradient only when a conditioning parameter trains (full finetune, `+modulation`
  adapters), so the only gradient a LoRA run computes and sends back across a boundary
  is the one of `x`.
- The micro-batch index reaches every stage as a keyword argument of the schedule
  (`kwarg_mbs`), and the last stage's loss takes it as its target (`target_mbs`), so
  every stage takes the index from the host; the schedule runs with
  `return_outputs=False` (outputs are not merged).
- Each micro-batch's adapter step state (T-LoRA masks, DyLoRA ranks, module-dropout
  draws) is drawn once per step in micro-batch order from its padded layout and
  installed again before each of its chunk forwards and backwards
  (`TimedStage.on_chunk`), so its checkpoint recompute sees the state its forward saw.

### Static shapes

Every micro-batch is padded to the pipe's token budget and sample count:

- The padding is split into zero-weight samples of at most `PipeShape.pad_segment`
  (512) tokens, so the padding's attention cost stays small, and the sample rows of the
  modulation are `MAX_SAMPLES + ceil(MAX_TOKENS / 512)`.
- `PipeShape.from_knobs` takes the shape from the data knobs (`MAX_TOKENS` joint tokens,
  `MAX_SAMPLES` samples, `MICROBATCHES`) and refuses knobs under which they are not hard
  bounds of the planner. It needs `BATCHING='tokens'` or `'tokens_ffd'`,
  `MAX_SAMPLES >= 1`, `TOKEN_BUDGET='exact'` and `TEXT_CACHE='offline'`.
- The padding share of the tokens is reported (`pipe_pad_frac`); first-fit packing
  (`tokens_ffd`) and a `MAX_TOKENS` that holds a whole number of typical samples keep
  it low.
- Token drop (`OVERLONG='drop_tokens'`) passes each sample's kept grid indices as
  `MicroBatch.img_keep`; the kept tokens keep their grid positions.

### Stage split

`PIPELINE_SPLIT` (default `"measured"`) chooses the layer ranges
(`kohakumoss.parallel.pipeline.split.stage_bounds`):

| value | costs |
| --- | --- |
| `"measured"` | rank 0 times forward + backward of every block (through the run's checkpointing runner), the prologue (conditioning, embedders, text-refiner blocks) and the epilogue (head + loss) at a micro-batch of typical samples filling the shape; the costs are broadcast so every rank builds the same stages |
| `"flops"` | an analytic FLOP model of the same three parts |
| `"even"` | every block costs the same, no prologue / epilogue |
| `(n0, n1, ...)` | explicit layers per virtual stage |

- The partition is an exact DP over contiguous cut points that minimises the slowest
  stage's cost (its layers plus the prologue on the first stage and the epilogue on the
  last) divided by its rank's speed; ties break on the sum of squared per-stage
  parameter counts (memory).
- Ranks on identical devices have speed 1; mixed devices use their measured dense-matmul
  rate.
- A V placement first splits per rank (rank 0 carries prologue and epilogue) and then
  halves each rank's layers between its two stages.
- The split table is logged at startup.

## Threaded pipeline

`ThreadedPipeline` places layer ranges on devices of one process:

- Kernel launches are asynchronous per device, so stages overlap from one thread;
  `schedule="async"` (`PIPELINE_SCHEDULE`, `"gpipe"` or `"async"`) runs each
  micro-batch's backward on a helper thread while the next forward is issued.
- Each stage can bind its own op set (`opset_for(device)`), so a 3090 stage runs Triton
  kernels and a 2080 Ti stage runs TileLang kernels.

## Measuring a run

### Step breakdown

Each engine records its part of the step on device events, read once per metric row:

| key | engine | what |
| --- | --- | --- |
| `ddp_comm_wait_ms` | ddp | the compute stream waiting for the gradient all-reduces after backward (exposed communication) |
| `ddp_span_ms` | ddp | the last micro-batch's forward + backward + that wait |
| `ddp_reduce_mb` | ddp | megabytes all-reduced per step |
| `pipe_fwd_ms`, `pipe_bwd_ms`, `pipe_loss_ms` | pipeline | this rank's stage compute per step |
| `pipe_wall_ms`, `pipe_idle_ms`, `pipe_bubble_frac` | pipeline | the schedule's span on this rank, the part not computing (bubble plus exposed transfers) and its fraction |
| `pipe_pad_frac` | pipeline | padding share of the micro-batch tokens |

### Profiling

| knob | effect |
| --- | --- |
| `TRACE_STEPS = (first, last)` | profiles steps `first + 1 .. last` on every rank with `torch.profiler` (any engine, single GPU too) and writes the chrome trace `<output>/ranks/trace.rank<r>.json.gz` |
| `PROFILE_STEPS = (first, last)` | calls `cudaProfilerStart` at the end of step `first` and `cudaProfilerStop` at the end of step `last` (CUDA devices), so an external profiler started with `ncu --profile-from-start off` or `nsys --capture-range=cudaProfilerApi` records steps `first + 1 .. last` only |

The `TRACE_STEPS` kernel timeline gives, per step:

- `trace_compute_ms` (union of compute kernels), `trace_comm_ms` (union of NCCL
  kernels), `trace_comm_exposed_ms` (NCCL time with no compute kernel running),
  `trace_copy_ms`, `trace_idle_ms` (no kernel or copy running) and the busy / compute
  fractions.
- `trace_phase_<p>_ms` for `p` in data, forward, backward, recompute, optimizer and
  other: the union of the compute kernels launched inside that step phase. The drivers
  open a profiler range per micro-batch around the host-to-device batch (`data`), the
  forward and the backward, and one around the optimiser step; every checkpoint
  recompute runs in a `recompute` range (`kohakumoss.logging.phases`). A kernel belongs
  to the innermost range open, on any host thread, when it was launched, so the
  autograd thread's kernels count as backward and a recompute inside backward counts
  as recompute. A kernel launched outside every range is `other`.
- `<output>/ranks/trace.rank<r>.kernels.json` lists, per phase, the 40 kernels with the
  most device time (ms and calls per step).
- Under the pipelines an NCCL receive kernel spins while it waits for its peer, so there
  `trace_comm_ms` includes waiting; `1 - trace_compute_frac` is the bubble. Under DDP the
  same spin counts the wait for the slowest rank.

The `TRACE_STEPS` steps run slower (CUPTI tracing). The steps of either window are left
out of the steady-state summary.

### Per-rank report

Every rank writes one row per metric step (`METRICS_EVERY`) to
`<output>/ranks/rank<r>.jsonl`:

- samples/s and tokens/s of the data it saw, step time;
- TFLOP/s and MFU of its share of the model (its stages under the pipeline, the whole
  model under data parallelism; per device under `threaded`);
- peak VRAM allocated / reserved since the previous row
  (`kohakumoss_kernels.hardware.memory` windows), device memory in use, `nvidia-smi`
  peak memory and mean utilization of its device (a background `nvidia-smi -lms 200`
  sampler);
- the engine's breakdown and the step's other metrics.

At the last step every rank's steady-state summary (rows after step 3, outside the
profiled window) is gathered and rank 0 writes `<output>/ranks/summary.json`: per rank
and for the job (samples/s summed over the data-parallel ranks, the slowest rank's step
time, peaks over ranks).

### Scaling runs

`configs/moss/scale/` holds the scaling layers:

| layer | setup |
| --- | --- |
| `scale_lora_bf16` | LoRA dim 32 on the static micro-batch knobs: ~1024^2 samples, 9216-token first-fit batches of at most 4 samples, offline caches, `ckpt-resident`, 30 steps, one metric row per step, steps 26-28 profiled |
| `scale_lora_fp16` | the same for V100 / 2080 Ti |
| `scale_full_fp16` | full finetune for `fsdp` |
| `scale_full_fp16_offload` | its single-GPU reference |

A host layer adds WEIGHTS, DATASETS, CACHE_DIR and OUTPUT_DIR; each run sets PARALLEL,
DEVICES, GRAD_ACC, MICROBATCHES and PIPELINE_SCHEDULE with `--set` and a RUN_NAME of its
own.

`kogine run scripts/dist/scaling_report.py --set ROOT=<output dir> --set HOST=<label>`
reads every run under ROOT and writes `scaling.json` / `scaling.md`. Per engine x GPU
count x precision x algorithm:

- the job samples/s and the scaling efficiency against the single-GPU run of the same
  settings;
- the step breakdown and peak VRAM;
- the loss parity of the first 20 steps against the single-GPU run that sees the same
  global batch (`GRAD_ACC` = data ranks x `GRAD_ACC`, or `MICROBATCHES` for the
  pipelines).

The pipelines draw rank 0's data and noise, so their parity reference is exact. Data
parallel ranks draw their own sigma and noise streams, so a DDP / FSDP2 run matches the
single-GPU run on data but not on noise, and its parity is statistical (FSDP2 against
DDP at the same world size is exact).

## Heterogeneous devices (library API)

- `balance.measure_speed` times a dense matmul per device; `split.rank_speeds` gathers
  every rank's speed when the devices differ.
- `split.partition` finds the contiguous stage split minimising the slowest stage's
  cost / speed.

A mixed 3090 + 2080 Ti job runs fp16, since the 2080 Ti has no bf16.

## The `PARALLEL` slot

`PARALLEL` is a registry slot (`kohakumoss.training.layouts.LAYOUTS`): each kind is a
`Layout` that holds every per-kind decision of a run, resolved once at the start:

| field | decides |
| --- | --- |
| `spawns` | one process per entry of `DEVICES` (ddp / fsdp / pipeline) or one process (none / threaded) |
| `shared_data` | pipeline: every rank draws rank 0's batches, sigma and noise (one data rank); the other kinds: each rank draws its own share, the data split over every rank |
| `pipelined` | a step is `MICROBATCHES` micro-batches through a pipeline engine instead of `GRAD_ACC` |
| `stage_rng` | checkpoints keep the RNG of every stage device (threaded) |
| `begin` | per-run setup before the build (pipeline: the retained-backward patch for the schedule) |
| `hooks` | the loop's reduce / agree / norm / accumulate hooks (`training.build.parallel`) |
| `stages` | the pipeline engine, its objective and this rank's trainable names and parameters |
| `parts` | the checkpoint parts (FSDP2 full state dicts, per-stage pipeline parts, sharded host optimizer state) |
| `driver` | the train loop: `DRIVER` (native / lightning) or the threaded / pipeline drivers |
| `full_weights` | the context in which every rank holds full weights for export (FSDP2: unsharded) |
| `problems` | the layout's knob problems (`problems(k)`), reported by knob validation |
| `plan_problems` | the layout's memory-plan problems (`plan_problems(k, plan)`), reported by knob validation |

A dotted path in `PARALLEL` names a `Layout` defined elsewhere; the knob checks that
name a kind (`'ddp'`, `'fsdp'`, ...) do not apply to it.
