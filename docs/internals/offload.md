# Memory profiles and offload

`kohakumoss.memory` turns a profile name into a `MemoryPlan` once, and
`apply_to_backbone` (training) / `apply_inference_plan` (`MossPipeline.from_pretrained
(memory=...)`, `MEMORY_PROFILE` in `sample.py`) install the engines the plan names. The training
loop then calls one composed `BlockRunner`; nothing branches on the plan per step. The
text encoder has its own `TextEncoderPlan`, placed by `apply_to_text_encoder` (section
"Text encoder" below and [text-encoder.md](text-encoder.md)).

## Profiles

| profile | checkpoint | activation offload | weight stream | optimiser state | fused step | gradient offload |
| --- | --- | --- | --- | --- | --- | --- |
| `speed` | none | no | off | GPU | no | no |
| `balanced` | block | no | ring (16 resident) | GPU | no | no |
| `low` | block | yes | ring (4 resident) | host, streamed to the GPU | yes | no |
| `minimal` | block | yes | ring (2 resident) | host, streamed to the GPU | yes | yes |
| `auto` | the fastest of the above whose estimate fits the device's free memory (see [Memory model](#memory-model)) | | | | | |

No profile sets the base precision: every profile keeps the compute dtype, and
`BASE_PRECISION` (`compute`, `int8`, `fp8`, `quant`, `fp8_compute`, `mxfp8`) applies under
any profile. The catalog entries (`ckpt-stream4-actoff-optstream`, ...) are every
combination of the placement axes; [../concepts/profiles.md](../concepts/profiles.md)
lists them with their estimates.

`optim_state`: `gpu` (fp32 masters and moments resident), `stream` (held in pinned host
memory, streamed through the GPU per parameter / per block; the update runs on the
GPU), `cpu` (`CPUAdamW`, update on the host). It is placement only: 8-bit states are
an `OPTIMIZER` (`adamw8bit`, `adamw8bit_kernel`) with GPU placement.

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
| `StageRunner` | global block index for a pipeline stage (the inner runner binds at the global indices) |

## Activation offload

`ActivationOffload` installs `saved_tensors_hooks` per block. A saved tensor above
`min_bytes` is copied device-to-host on a side stream during forward (its storage stays
reserved for that stream until the copy finishes). In backward the first unpack of block
`i` fetches all of block `i`'s packets and prefetches block `i-1` on a second stream. The
device buffers of a fetch are allocated on the consuming (compute) stream, and the copy
stream first waits for the compute stream. Freed buffers then return to the compute
stream's allocator pool, the pool that block `i-1`'s own work allocates from. A buffer
allocated under the copy stream would sit in that stream's pool, which the compute stream
never draws from, so every step would reserve the offloaded bytes a second time.
Parameters, buffers and streamed weight slots are excluded by storage pointer. The
excluded set is rebuilt from the live backbone tensors, ring slots and trainable
parameters once the build is complete (`MemoryEngines.exclude_params`), because the
build replaces storages after the plan is applied (full FT casts trainable parameters to
fp32): a pointer of a freed bf16 weight would otherwise stay excluded, and an activation
the allocator later places at that address would never be offloaded. Combined with block
checkpointing, the checkpoint's saved block inputs are the offloaded tensors.

Two saved tensors are not copied. A saved dtype cast of a leaf parameter, or a view of
one (autocast's weight cast under a full-FT forward without checkpointing, saved as
`w.t()` by the linear: view nodes over `ToCopyBackward0` over the parameter's
`AccumulateGrad`), is recorded as the parameter, the dtype and the view's shape /
stride / offset, and cast and viewed again on unpack; copying it would move a 16-bit
copy of every weight D2H and back each step. The
unpack checks the parameter's version counter and raises if it changed in place since
the forward. A tensor saved twice in one block (a streamed linear and a LyCORIS bypass
module both save the linear's input) is one packet with two users, keyed by storage,
offset, shape, stride, dtype and version, and only while the first saved tensor is
alive (a weak reference), so a new tensor at a freed address never matches.

A device tensor handed to a copy stays reserved until the copy has run on the device
(`record_stream`). Nothing else holds the host back (the ring weight stream takes no
host sync per block), so an unbounded host queues the whole forward before the device
runs it and every block's offloaded input is reserved at once.
`act_offload_inflight` (default 2) bounds it: each direction keeps the latest copy event
of its newest blocks, and a copy of a block beyond that count first waits on the host
(`Event.synchronize`) for the oldest block's last copy. The host then runs at most that
many blocks ahead of the device, which still leaves the device that many blocks of
queued work. The memory estimate counts `max(act_offload_inflight, 2)` blocks of saved
activations (a backward block and its prefetched predecessor).

RTX 5090, LoRA, online VAE and text encode, 1 MP, 6 steps
(`tests/memory/real/test_profile_memory.py`), torch allocator counters per step:

| profile | allocated peak GiB | reserved peak GiB, paced | reserved peak GiB, unpaced | tokens / s, paced | tokens / s, unpaced |
| --- | --- | --- | --- | --- | --- |
| `minimal` | 3.96 | 4.45 | 7.08 | 3034 | 3015 |
| `low` | 4.38 | 4.95 | 7.71 | 2981 | 2958 |

Unpaced, 2.79 GiB of `minimal`'s reserved peak are block inputs whose free waits for
their device-to-host copy.

Host buffers come from a pool of pinned byte buffers in power-of-two size classes (at
least 1 MiB). A packet takes the smallest free class that holds it, so a smaller bucket
shape reuses a larger buffer. A buffer returns to the pool when its H2D copy is queued,
and it is reused only after that copy has finished. New pinned memory is allocated only
when a packet is larger than every free buffer. A pinned allocation (`cudaHostAlloc`)
blocks the host for about 10-30 ms per 8-32 MiB buffer on the V100 host, and 42 of them
per step cost more than the step's compute. With the pool, the LoRA `balanced` step on
V100 with offload takes 1207 ms against 1199 ms without it.

## Weight streaming

`WeightStream` keeps every streamed `MossLinear` weight in (pinned) host memory and a ring
of `resident` device slots. `enter(i)` makes block `i` resident and prefetches `i+1`;
backward makes `i` resident again and prefetches `i-1`. A slot is overwritten only after
the compute stream records that it is done with the slot's previous block. A streamed
linear (`_StreamedLinear`) saves only its input for the backward, never the slot. Its
backward reads the weight from the slot again: the block's backward makes the block
resident (`_Exit`), which re-streams it if a later block took the slot. So a streamed
block needs no checkpointing. Without checkpointing its activations stay saved and the
recompute is skipped. The backward re-streams the block's weights once, as the
checkpointed recompute does. On a trainable block the weight gradient is `g^T x` from
the saved input. An fp8-compute base (`fp8_compute`, `mxfp8`) streams through
`_StreamedFp8Linear`, which saves nothing: the forward is the gradient-free
`fp8_linear_infer`, and the input gradient (fp8 through the streamed `q_t` copy, or
16-bit through `q`) reads the parts from the slot again through `fp8_linear`'s backend
launcher (`kohakumoss_kernels.fp8.op.dgrad`). The base is frozen, so no input is kept.

`WeightStream` (`offload/weights/stream.py`) owns the host copies (`host`, the trainable fp32
masters `params`, the slot `layout`) and installs the streamed bases; the work is done
by five parts in `offload/weights/streaming/`, built once per stream for its device:

| part | job |
| --- | --- |
| `SlotRing` (`ring`) | the device slots, which block each holds, the LRU victim that skips the active and kept blocks; `ensure` / `prefetch` / `release` |
| `H2DCopier` (`copier`) | host -> slot copies on the H2D side stream (pinned, or through the two staging sets), the per-slot `ready` / `free` and per-block `host_ready` events; a plain copy on the host |
| `SpanTimer` (`timer`) | copy and copy-wait spans while timing is on, cut per step (`set_timing`, `cut_timing`, `read_timing`) |
| `GradSink` (`grads`) | the fp32 host gradient sums (`host_grad`, `filled`), the D2H through `HostAccumulator`, `.grad` exposure for `flush_grads` |
| `BlockReducer` (`reducer`) | the last micro-batch's cross-rank block sum: the one fp32 flat buffer, the side stream, `reduce` |

The device-dependent calls (`copier.copy` / `wait` / `mark_free`,
`grads.accumulate`) are bound when the part is built, and the streamed weights call
`slot` / `ensure` / `prefetch` / `release` as the ring's own bound methods. The parts
share one `stats` dict (loads, H2D / D2H bytes, host syncs).

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
  next forward's H2D of block `i` waits on that write-back. A slot holds per block
  parameter an fp32 gradient and the streaming-dtype weight, plus the fp32 master and
  moments rows for streamed states (`slot_bytes` in `offload/optim/stream_optim.py`, which the
  memory estimate uses): 3 x 18 B per parameter in bf16 with resident states, 3 x 54 B
  with streamed states. A block whose slot is needed again before its last gradient
  arrived, or that is never stepped in the backward, raises instead of having its state
  rows overwritten. The host gradient sums are never zeroed: a step's first gradient
  of a weight is copied into its sum and later ones added (`grad_filled` marks the sums
  that hold this step's gradients). With `optim_state="cpu"`
  the sums land in the masters' `.grad` (`flush_grads()`), `CPUAdamW` steps, and
  `refresh()` updates the streaming copies; under DDP each block's last-micro-batch
  gradient plus the host sum becomes one fp32 buffer on a side stream that is reduced
  onto the block's owner and copied to its host gradient (`BlockReducer.flush_block`). Under DDP
  the masters and streaming copies
  are one shared host region for all ranks of the node (see
  [multi-gpu.md](../guides/multi-gpu.md#shared-host-masters)).
- **Pinned vs staged**: `pinned=False` keeps pageable host copies and copies each block
  through two alternating pinned staging sets (one buffer per weight each): before
  refilling a set the host waits only for the H2D that last read it, two block loads
  back, so the host memcpy of block `i+1` overlaps the H2D of block `i`; a trainable
  ring also waits for the block's master write-back.

## Allocator

Allocated memory of a run is flat from step to step, but the token count of a packed
batch varies, so the caching allocator receives a different set of block sizes every
step. With torch's default settings it splits cached segments to fit them and reserves
new segments when no split fits, and reserved memory (what nvidia-smi shows) grows over
the run while allocated memory does not. Three parts keep reserved memory near the
allocated peak and steady from step to step:

- **Expandable segments** (training and `sample.py`, applied by `configure_allocator`
  in `offload/accel.py` before the run's first allocation): `expandable_segments:True`
  where the CUDA allocator has them (Linux; segments grow and shrink in place, so a new
  size never needs a new segment), torch's settings elsewhere. A
  `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF` in the environment wins. The
  settings in force are logged at the start of the run.
- **No per-step allocation on a side stream.** A block allocated while a side stream is
  current belongs to that stream's pool, which the compute stream never draws from. The
  offload engines therefore allocate their device buffers once and copy into them:
  the weight ring slots (`WeightStream`), the optimiser-state rings (`HostStateAdamW`:
  master, moments and the fp32 gradient in one 4-row slot; `StreamedAdamW`: the same
  four rows plus the updated weight in the streaming dtype, the gradient and weight only
  with resident states), the gradient-offload
  staging ring (`GradOffload`), the fp32 flat buffer a host-stepped streamed block's
  gradients are summed across ranks in (`WeightStream.reduce_with`, allocated once on
  the compute stream), and the activation-offload fetch buffers, which are allocated
  on the compute stream (see Activation offload).
- **No release between steps** (below).

### Between-step release

The trainer returns no memory to the driver between steps. The caching allocator keeps
the blocks a step freed and hands them to the next step's allocations of the same
stream; the offload engines copy into the fixed buffers above, so a step neither frees
nor re-reserves what the previous step used, and reserved memory stays at the steady
peak instead of being unmapped and mapped again every step. Memory goes back to the
driver only at phase boundaries: after the build, after the kernel and adapter
prewarm and the text-encoder prewarm, and around sampling (`MossPipeline` releases
after a generate); the between-step retune of the first steps returns nothing (only a
queued problem whose inputs do not fit releases before it is dropped), and no step
after the tuning windows tunes (docs/internals/tuning.md#tuning-windows). Under memory pressure the allocator itself frees cached blocks and retries an
allocation before it reports out-of-memory. `STEP_RELEASE` (default `"none"`) is the
only explicit release inside the loop: `"empty_cache"` (every optimizer step) or N
(every N steps) are opt-ins, each a device synchronise plus `empty_cache` that the next
step pays back as fresh allocations.

`kohakumoss.memory.release.CacheReleaseStats` runs after every step on CUDA /
XPU and reports the bytes returned to the driver at that step, as reserved bytes
before minus reserved bytes after. With a STEP_RELEASE callback bound, these are its
release's (`memory_reserved` read around the synchronise + empty_cache, 0 on a step
it does not run); without one, they are the drop of the allocator's reserved bytes
since the previous step end. It frees nothing. The allocator's
`reserved_bytes.all.freed` counter is not used: it counts every segment freed and
re-reserved inside a step, so its per-step difference exceeds the device's memory
under churn. The metric rows carry `cache_released_gib` (GiB returned at that step)
and, averaged since the previous row, `cache_releases` (share of steps in which memory
went back) and `cache_release_time` (host seconds per step of the release). With no
release bound a steady run shows zeros, and a nonzero row marks an allocation retry or
a phase boundary inside that step (a sample grid, the text-encoder prewarm before the
first encode).

A streamed 16-bit linear reads its slot in the autocast dtype when autocast is on, so a
call never materializes a full-precision copy of the weight.

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
  registration order). `finish()` makes the compute stream wait for the updates. The
  caller's gradient and finite flag are read on the update stream, so they are recorded
  on it (`record_stream`): the allocator does not reuse their memory before the update
  has read them.
- `StreamedAdamW`: per-block GPU AdamW for `swap` blocks (see Weight streaming).
- `CPUAdamW` is a `torch.optim.Optimizer` whose fp32 masters and moments live on the host;
  gradients go device -> pinned host, the update uses `torch._fused_adamw_` on the CPU, and
  updated weights go back host -> device. `step_param(p)` updates one parameter. For a
  device parameter it queues the gradient and the finite flag device -> pinned host on a
  side stream and returns. A host worker thread waits for that copy, runs the update and
  queues the host -> device copy of the new weight on a second side stream. `finish()`
  joins the worker and makes the compute stream wait for the copies. The host update
  therefore overlaps the rest of backward. A 16-bit host gradient is converted for the
  update into one reused fp32 host buffer (one for the worker thread, one for the
  caller's thread), not a new tensor per parameter and step; a parameter's pinned
  gradient buffer is allocated on its first D2H, so parameters that read gradient
  offload's host sums (`use_host_grads`) never hold one. `load_state_dict` keeps the
  masters and moments as host fp32 tensors (torch's loader casts optimiser state to each
  parameter's device and dtype) and rebinds each parameter to its reloaded group.
- All four take torch-style parameter groups; each group's lr / betas / eps /
  weight_decay apply to its parameters, and `param_groups` holds the live dicts a
  schedule edits. They apply `decay_to_one` groups to their fp32 masters themselves
  (`handles_decay_to_one`), so the bundle's post-step decay is skipped for them.
- `FusedStep` registers post-accumulate-grad hooks that step each parameter as soon as its
  gradient is complete and free it, so only one block's gradients are alive.
  `PerParamOptimizer` gives any torch optimiser that per-parameter interface behind a
  torch-style `param_groups` view for schedules (one view dict per source group).
  `FusedStepBundle.step` marks the schedule's optimiser as stepped (`_opt_called`)
  before `schedule.step()`: the in-backward updates are that optimiser's step, and
  torch's LRScheduler otherwise warns that the schedule stepped before the optimiser.

### Fused step and non-finite gradients

`FusedStepBundle.step_param(p)` runs one transform per gradient and passes its flag to
the update as `found_inf`. The transform makes two passes over the gradient: the fp32
norm of the scaled gradient (its unscaled square goes into the step's norm), then
`torch._amp_foreach_non_finite_check_and_unscale_` with the factor `inv_scale x
min(1, GRAD_CLIP / G)`, which unscales and clips in place and sets the device-side flag
on a non-finite value. `G` is the global gradient norm of the previous step (computed
on the device at its `step()`, kept across a non-finite step, saved in the
checkpoint): `GRAD_CLIP` is the global norm in both bundles, and the fused step's own
global norm exists only after every parameter has already stepped. A run's first step
is not clipped. A flagged gradient keeps its values: every
update path below leaves the parameter and its state untouched where the flag is set,
and the step's norm is reported as NaN. Every update path honours the flag:

- the offload AdamWs (`MasterAdamW`, `HostStateAdamW`, `StreamedAdamW`) hand it to
  `torch._fused_adamw_`, which leaves the master, both moments and the step count
  untouched when it is set; `CPUAdamW` skips the update;
- a per-parameter torch optimiser built with `fused=True` gets it as `found_inf`
  (parameter, moments and the device step count untouched);
- `AdamW16` and `MuonHybrid` run `step_masked(found)`: the update is computed into
  working copies and each parameter / state tensor is stored through
  `torch.where(found, old, new)`;
- every other per-parameter optimiser (`adamw` off CUDA, `adamw8bit_kernel`,
  `prodigy`, `schedule_free`) runs its own step between a snapshot and a device-side put-back:
  the parameter and every state tensor on its device are cloned before the step and
  restored through `torch.where(found, old, new)` after it. The clones are transient,
  one parameter at a time (its value plus its state), and cost one extra read and write
  of each per update.
- bitsandbytes optimisers (`OPTIMIZER='adamw8bit'`) run their per-parameter step body
  (`init_state`, `prefetch_state`, `update_step`) directly, inside that snapshot. Their
  `step()` calls `torch.cuda.synchronize()` after every parameter, which would stall
  backward once per parameter. On V100, LoRA `balanced` with 8-bit states and
  `fused_step` takes 1575 ms this way and 2629 ms through `step()`; `low` takes 1336 ms
  and 6658 ms.

Optimiser state the device cannot mask without a host sync (Python step counts, step
counts held as host tensors, group values such as Prodigy's `d` and `k`) is recorded per
parameter before its update (`training/optim/optimizers/skip.py`, `host_state`). `step()` already
syncs once to read the step's finite flag; on a non-finite step it reads the
per-parameter flags in one more sync and puts that record back for every flagged
parameter, and a state the skipped update created is dropped. Finite steps read nothing
more. A skipped parameter therefore ends the step with its value and its whole optimiser
state as before it, the same as under `torch._fused_adamw_`.

The non-fused bundle (`training/optim/bundle/base.py`) reads the agreed finite flag before
`optimizer.step()` and skips the whole step on a non-finite one, so no parameter or
state moves there.

A parameter whose gradient is non-finite is therefore not updated; every other parameter
of that step was already updated in backward and stays updated. The scaler backs off and
the step is reported non-finite: its `grad_norm` is NaN (as in the non-fused path, the
norm is not taken over the remaining finite gradients), `nonfinite_params` counts the
flagged parameters, and the instability dump names the first of them. A non-finite
gradient that repeats would otherwise leave those parameters frozen while the run goes
on, so `NONFINITE_PATIENCE` (default 8, 0 = off) stops the run with their names after
that many consecutive non-finite steps at the loss-scale floor (every non-finite step
of a run without a dynamic scaler is at the floor), or after the same parameters are
non-finite that many steps in a row.

`step()` first runs the bundle's finishers: the compute stream waits for every side
stream that updated parameters inside backward (`GradOffload.finish`,
`HostStateAdamW.finish`, `StreamedAdamW.finish`), before the finite flag is read and
before the next forward reads the parameters.

### Wiring (`training/memory/optimizers.py`)

`build_memory_optim` is the one place that turns a plan into optimisers, for the trainer
(`build_optim`) and for `MemorySession` alike:

| plan | device parameters | `swap` block masters | step |
| --- | --- | --- | --- |
| `optim_state='gpu'` | `adamw` (torch AdamW, also in MemorySession by default) | `StreamedAdamW`, resident states | in backward for `swap`, `fused_step` or `grad_offload`; else at `step()` |
| `optim_state='stream'` | `HostStateAdamW` | `StreamedAdamW`, streamed states | same |
| `optim_state='cpu'` | `CPUAdamW` | `CPUAdamW` after `flush_grads` | at `step()`; with `fused_step`, device parameters in backward |

`resolve` sets `fused_step` for `swap` with a GPU optim_state (blocks can only step inside
backward) and for `grad_offload` with a GPU optim_state (otherwise every summed gradient
comes back to the device before the step, and the peak holds them all), unless
`fused_step` is overridden; `swap` with a contradicting override is rejected.
`grad_offload` + `fused_step` + `CPUAdamW` finishes each gradient on the device and the
host update reads it from there (one extra round trip per parameter); without
`fused_step` the host update reads the host sums directly.

## Host workers

`HostAccumulator` (gradient sums) and `CPUAdamW` (host updates) each run their host-side
jobs on one `HostWorker` thread. `HostAccumulator` copies each device gradient into the
next buffer of a ring of `STAGING_SLOTS` (4) pinned staging buffers shared by every
parameter, each grown to the largest gradient it took, and the worker adds it into the
fp32 sum; a buffer is reused once its add has landed, so backward waits only when the
worker is four gradients behind. Its host staging is 4 x the largest gradient in the
gradient's dtype, independent of the parameter count: under `swap` with `GRAD_ACC > 1`
the host holds the fp32 sums (4 B per trainable block parameter) and nothing per
parameter beyond them. The thread holds only its queue, never its owner. It
stops on `close()` (queued jobs run first), or when the owner is released, through a weak
finalizer that posts a stop sentinel and joins with a timeout. The trainer closes the
memory engines and the optimisers when a run ends (`close_memory`), so repeated runs in
one process (notebooks, sweeps, test sessions) do not accumulate threads.

## Gradient offload with accumulation

`GradOffload` registers a post-accumulate-grad hook per device parameter: as soon as
autograd has the micro-batch's gradient it is copied device -> pinned host on a side
stream and the device gradient is freed; a host worker thread (`HostAccumulator`) keeps
an fp32 sum across the `GRAD_ACC` micro-batches. No full-size gradient buffer stays on
the device. On the last micro-batch (`start(last=True, final=opt.step_param)`) each
parameter's host sum comes back through a 3-buffer device ring on an H2D stream, is
added to the fresh device gradient on an update stream, and the GPU optimiser steps
that parameter right there; `finish()` makes the compute stream wait for the updates. A
host optimiser reads the sums directly instead (`CPUAdamW.use_host_grads`; only the
sums that hold this step's gradients). Streamed `swap` blocks accumulate on the host
through `WeightStream`.

A step's first gradient of a parameter is copied into its host sum and later ones are
added, so no sum is zeroed between steps (no multi-GB host memset on the main thread);
`filled` marks the sums holding this step's gradients, and a parameter with none gets no
gradient (`restore` leaves `.grad` None, as without offload). A parameter whose hook does
not fire on the last micro-batch but whose sum holds earlier micro-batches' gradients
(module dropout, a branch a batch skips) is finished by `settle()` after that backward
(its sum becomes its gradient and goes through the same final callback), before the DDP
reduction ends. Under DDP a gradient absent on some ranks counts as zero there; one
absent on every rank is not delivered (each bucket carries one presence count per
parameter in its all-reduce, read on the host only when a parameter is missing
locally), so it is not stepped, as on one GPU.

## MemorySession and parameter groups

`MemorySession(model, plan, device, dtype, grad_acc, trainable_base, ...,
param_groups)` applies the plan (`apply_to_backbone`) and builds its bundle through the
trainer's `build_memory_optim`, for benches and tests on models outside the trainer. The driver calls `backward(loss, micro)` per
micro-batch and `step()` once per optimiser step; `masters(model)` returns every fp32
master by name.

`param_groups` are torch-style groups over the model's parameters as given before the
session is built (parameters the plan replaces, such as streamed block weights, are
followed by name). Each group is split into its device part and its streamed part, and
both parts keep the group's hyperparameters and extra keys, so per-group rules (no
weight decay on biases / norms / 1-D parameters, decay-to-one, LoRA+ ratios) reach
every optimiser path. Every trainable parameter must be in a group.

## Rejected combinations

Combinations the knobs reject (`swap` with a contradicting `fused_step`) are rejected by
`validate`; invalid plan combinations by `resolve`. A quantized base cannot be trained
(full FT).

## Hot paths

Per-block, per-tensor and per-parameter paths are specialised at build time: runners
`bind(layers)` to a tuple of per-block callables, the offload / stream / gradient
engines bind their device-or-host copy and wait functions at construction, `start()`
swaps the hook body once per micro-batch, and `QuantWeight.bind` sets `linear` to the
kernel or the dequantizing path. Per-step and build-time choices stay plain `if`s.

## Quantized base

`quantize_model` replaces accepted `MossLinear` bases with `QuantWeight` (per-output-row
scales: int8 `round(w / s)`, `s = max|w_row| / 127`; fp8 e4m3 `w / s`, `s = max|w_row| / 448`).
`QuantWeight.bind` resolves a `quant_linear` kernel for the device once; without one the
weight is dequantized per call. Every `quant_linear` entry, eager included, is the same
custom op (`kohakumoss_kernels.quant.make_quant_linear`), whose autograd saves `q` and
`scale` and dequantizes again for the input gradient, so a frozen int8 / fp8 base never
keeps a 16-bit copy of a weight from forward to backward. `split_quantized` / `attach_quantized` load ComfyUI fp8 /
int8 checkpoints (`scale_weight` or `weight_scale` keys). The text encoder's linears use
`kohakumoss.quant.text` (per channel or per group, weight-only or fp8 compute).

## Text encoder

`apply_to_text_encoder(text_model, TextEncoderPlan, device, dtype)` is the one entry for
training (`build_encoders`, the online text cache, sample prompts), `MossPipeline` and
the offline cache factory. It quantizes the linears (`precision`), resolves
`memory="auto"`, then places the model (`kohakumoss.offload.encoders.text`):

| memory | between passes on the device | during a pass | host-to-device per pass |
| --- | --- | --- | --- |
| `resident` | everything | everything | 0 |
| `offload` | nothing | everything but the embedding table | all weights (pinned copies) |
| `stream` | final norm, rotary tables | + a ring of `stream_resident` slots per layer kind | all layer weights, prefetched one layer ahead |
| `cpu` | nothing (runs on the host) | nothing | 0 |

Under `offload` / `stream` the 2 GiB embedding table never moves: the pass looks up
the rows on the host and sends only them. Every placement's `end()` releases the
pass's device weight buffers (the ring slots are freed, offloaded copies dropped)
before the caller's next work, so the DiT step never shares the device with them; the
VRAM timeline (`vram_te_peak`, `vram_te_after`, `vram_step_peak`) records it.
`LayerStreamer` swaps a layer's tensors into its slot on entry and back to the host
copy on exit, through forward hooks (transformers forward) or `enter(i)` / `leave(i)`
(the kernel forward, which runs layer-major over every pack of a pass so a streamed
layer crosses the bus once per pass); layers of different structure get separate slot
pools; `h2d_bytes` counts the copies.

`free_after_use` drops every weight after the first pass (the empty-prompt embedding is
encoded in it and kept). Training sets it when text is cached offline: the sample
prompts are encoded once at build and the encoder is freed.

The memory model adds the text encoder (`kohakumoss.offload.encoders.text_memory`): resident
weights count against the DiT step peak; an offload / stream pass peaks at the
persistent DiT parts plus the pass's weights and activations (all pack hidden states,
one call's MLP intermediates, the delta-rule scratch). `with_text` and
`auto_plan(text=...)` use it; `memory="auto"` picks the fastest placement that fits
next to the step: `resident`, else `stream`. `offload` is never picked: streaming
prefetches each layer while the previous one computes, so it is faster than the bulk
copy of `offload` and holds one ring instead of every weight (Qwen3.5-9B, 64 captions
per pass, packed, 32 per call: 3090 bf16 4328 vs 3777 tok/s, V100 fp16 4613 vs 2735;
pass peak 2.9 vs 14.2 GiB).

## VAE

`VAE_ENCODE` (`kohakumoss.offload.encoders.vae.VaePlan`: `memory`, `batch`, `ahead`, `tile`)
places the FLUX.2 VAE for online latent encoding and for sample decodes
(`PlacedVae`, built by `place_vae` in `build_encoders`):

| memory | between passes | during a pass |
| --- | --- | --- |
| `resident` (default) | weights on the device | weights + activations |
| `offload` | nothing (pinned host copy) | weights moved in, released after the pass |
| `stream` | nothing | encoder / decoder blocks through 2-slot rings, the rest moved in per pass |
| `cpu` | host | host (training: offline latent cache only, i.e. sample decodes) |
| `auto` | the fastest of resident / offload / stream that fits next to the DiT step reserve | |

`offload` copies the weights in from their pinned host copies on the compute stream at
the start of a pass and drops the device copies at its end. They were allocated and read
on that stream, so the allocator hands their memory to later work only in that stream's
order, and the pass ends without a device synchronize (as the text encoder's `offload`).
`vae_time` is then the host time of the pass, as for `resident`.

`tile` turns on the diffusers VAE's tiled and sliced encode / decode, so activation
peaks stay bounded at high resolution. With the online latent cache `LatentAhead`
encodes the cache misses of the next `ahead` host batches in one VAE pass and holds
the latents until the batcher takes them, so an offloaded VAE moves once per `ahead`
batches and is released before the DiT step. `VaeEncodeStats` logs `vae_time`,
`vae_share`, `vae_images` and `vae_h2d_gb` per step; the VRAM timeline marks the VAE
pass (`vram_vae_peak`, `vram_vae_after`) apart from the DiT step. The `8g` and `12g`
VRAM layers offload the VAE, `auto` picks.

### VAE dtype, compile and batching

`VAE_DTYPE` (training, `scripts/data/cache_latents.py`, `scripts/sample/sample.py`;
`kohakumoss.encoders.vae.resolve_vae_dtype`) selects the VAE's weight and compute dtype.
It is `auto` | `bf16` | `fp16` | `fp32`.

- `auto` is bf16 where the card has bf16 and fp16 elsewhere (V100, 2080 Ti); on the CPU it
  is fp32.
- One dtype covers every path: the online latent encode and the sample decode in
  training (both `PlacedVae`), the offline cache pass (`vae_encoder`) and the pipeline
  decode (`MossPipeline.from_pretrained(vae_dtype=...)`). Latents always come out in
  fp32.
- The latent normalisation (the VAE BatchNorm's 128 running means and variances) stays
  fp32 whatever the dtype: `FluxVae.from_pretrained` loads the VAE in fp32, keeps those
  statistics, then casts the VAE (diffusers' `torch_dtype` would round them to the VAE
  dtype, a per-channel shift and scale on every latent).
- The VAE dtype is part of the latent cache key (`LATENT_ENCODER_SPEC`,
  [data.md](../concepts/data.md#cache-keys)).
- The diffusers config sets `force_upcast: True`. Only diffusers' own pipelines read it,
  so KohakuMoss ignores it.
- fp16 is allowed because it stayed finite on real images (table below):
  `FP16_PROBLEM = None`. Setting `FP16_PROBLEM` to a reason rejects `VAE_DTYPE='fp16'`
  with that reason, and `auto` then never picks fp16.

Precision against the fp32 VAE: 16 dataset images, centre-cropped to 1 MP (1024²) and
2 MP (1440²). Encode is compared on the latents. Decode is compared on the pixels,
starting from the fp32 latents. Values are means over the images; "worst cos" is the
lowest image.

| GPU | dtype | area | pass | finite | cos (mean / worst) | abs mean | abs median | rel median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 3090 | bf16 | 1MP | encode | yes | 0.999887 / 0.999695 | 8.25e-03 | 5.66e-03 | 9.84e-03 |
| RTX 3090 | bf16 | 1MP | decode | yes | 0.999994 / 0.999988 | 1.96e-03 | 1.61e-03 | 3.22e-03 |
| RTX 3090 | fp16 | 1MP | encode | yes | 0.999983 / 0.999958 | 2.96e-03 | 1.75e-03 | 3.16e-03 |
| RTX 3090 | fp16 | 1MP | decode | yes | 1.000000 / 1.000000 | 2.12e-04 | 1.70e-04 | 3.42e-04 |
| RTX 3090 | bf16 | 2MP | encode | yes | 0.999870 / 0.999676 | 8.83e-03 | 6.05e-03 | 1.08e-02 |
| RTX 3090 | bf16 | 2MP | decode | yes | 0.999995 / 0.999990 | 1.89e-03 | 1.54e-03 | 3.13e-03 |
| RTX 3090 | fp16 | 2MP | encode | yes | 0.999980 / 0.999953 | 3.21e-03 | 1.91e-03 | 3.50e-03 |
| RTX 3090 | fp16 | 2MP | decode | yes | 1.000000 / 1.000000 | 2.12e-04 | 1.71e-04 | 3.47e-04 |
| RTX 5090 | bf16 | 1MP | encode | yes | 0.999887 / 0.999696 | 8.24e-03 | 5.64e-03 | 9.82e-03 |
| RTX 5090 | bf16 | 1MP | decode | yes | 0.999994 / 0.999988 | 1.97e-03 | 1.61e-03 | 3.23e-03 |
| RTX 5090 | fp16 | 1MP | encode | yes | 0.999983 / 0.999958 | 2.96e-03 | 1.74e-03 | 3.14e-03 |
| RTX 5090 | fp16 | 1MP | decode | yes | 1.000000 / 1.000000 | 2.22e-04 | 1.79e-04 | 3.58e-04 |
| RTX 5090 | bf16 | 2MP | encode | yes | 0.999870 / 0.999677 | 8.83e-03 | 6.05e-03 | 1.08e-02 |
| RTX 5090 | bf16 | 2MP | decode | yes | 0.999995 / 0.999990 | 1.88e-03 | 1.54e-03 | 3.13e-03 |
| RTX 5090 | fp16 | 2MP | encode | yes | 0.999980 / 0.999953 | 3.21e-03 | 1.91e-03 | 3.50e-03 |
| RTX 5090 | fp16 | 2MP | decode | yes | 1.000000 / 1.000000 | 2.12e-04 | 1.71e-04 | 3.47e-04 |
| V100 | fp16 | 1MP | encode | yes | 0.999989 / 0.999970 | 2.39e-03 | 1.48e-03 | 2.64e-03 |
| V100 | fp16 | 1MP | decode | yes | 1.000000 / 1.000000 | 2.18e-04 | 1.74e-04 | 3.51e-04 |
| V100 | fp16 | 2MP | encode | yes | 0.999987 / 0.999966 | 2.59e-03 | 1.60e-03 | 2.90e-03 |
| V100 | fp16 | 2MP | decode | yes | 1.000000 / 1.000000 | 2.09e-04 | 1.69e-04 | 3.43e-04 |

- Overflow: fp16 stays finite on every image at both areas, in both encode and decode.
  Its error is about 3x (encode) to 9x (decode) lower than bf16's, since fp16 has 10
  mantissa bits against 7.
- Speed: bf16 and fp16 run at the same speed. Both are 1.7-2.3x faster than fp32 and
  take about 60% of its VRAM.
- `auto` keeps bf16 on bf16 cards for its fp32-sized range. fp16 would buy precision
  there, not speed.

`VAE_ENCODE['compile']` (`auto` | `on` | `off`) runs encode and decode in channels_last
through `torch.compile(dynamic=True)` (`FluxVae.accelerate`). `auto` turns it on for a
resident CUDA VAE without tiling. The cache encoder and the pipeline decode compile on
CUDA too (the pipeline with a resident VAE only).

- The two are measured as a pair. channels_last alone is 40-75% slower than the
  default layout; compiled, it is the fastest variant at batch 1 on every GPU (bf16 at
  1 MP, per image):
  - RTX 5090: encode 39 ms against 59 eager, decode 53 against 79;
  - RTX 3090: encode 109 against 134, decode 153 against 177;
  - V100 (fp16): encode 137 against 143, decode 172 against 193.
- The compiled peak is lower too: 1.16 GiB against 1.65 at 1 MP encode.
- In fp32, compile ties or wins on the RTX 3090 and RTX 5090 but is 8-10% slower on the
  V100. With `VAE_DTYPE='fp32'` on a V100, set `compile='off'`. `auto` never picks fp32
  on CUDA.

Batching: `encode_images` encodes runs of same-size images together, at most
`VAE_ENCODE['batch']` images and `BATCH_PIXELS` pixels per call. The default is batch 1.

- Eager encode at batch 4 is 11-22% faster per image, but needs 4x the activations.
- Compiled encode is 10-18% slower per image at batch 4 than at batch 1.
- Decode never gains from batching.

`tile` changes nothing at 1 MP, because the tiles only engage above the diffusers tile
size. At 2 MP in 16-bit it holds the peak at 1.4-2.0 GiB, against 2.7-3.7 GiB untiled,
and costs 19-45% more time. The per-GPU speed table is in
[performance/vae.md](../performance/vae.md).

## Memory model

`estimate(Workload, MemoryPlan)` adds weights, trainable state, saved activations, the
recompute window, text-refiner activations, dequant transients and IO, times the plan's
fragmentation factor (`fragmentation(plan)`: 1.08, or 1.12 with side-stream buffers),
plus the 0.88 GiB outside the allocator ([profiles.md](../concepts/profiles.md#estimated-peak-vram)). The full-FT offload optimiser buffers are what the engines
allocate: `StreamedAdamW`'s slot ring for `swap` blocks (`stream_optim.slot_bytes`),
`HostStateAdamW`'s state ring under `optim_state='stream'` (`gpu_adamw.ring_bytes`)
and gradient offload's staging ring (`grad.ring_bytes`), both sized by the largest
device parameter (`largest_device_param`). Under FSDP2 the full-FT gradients count as sharded, except with
`GRAD_ACC > 1` and `FSDP_REDUCE='step'`, where every rank holds the unsharded gradients
of the whole model between the micro-batches (`Workload.sharded_grads`). EMA averages
count 4 B per trainable parameter per decay on the device (`Workload.ema_decays`);
under `swap` the block masters and their averages are host memory, so only the
device-resident parameters count. Saved activations per token per block are `12·D + 3·F` elements
without checkpointing and `D` (the block input) with block checkpointing; `act_factor`
scales the within-block part (1.0 for the fused kernel ops, 3.6 measured for the eager
ops). `auto_plan` walks profiles fastest-first and takes the first runnable one whose
estimate fits. The budget is the free memory at planning time times 0.92; free memory
already excludes the CUDA / XPU context, so each estimate (context included) is compared
with the budget plus its context part. The trainer and the pipeline plan from the
estimate alone.

Measured on a V100 32 GB (PCIe gen3 x8), MOSS-Image base, random weights, fp16, one
4096 + 300-token sample, norm + embedder weights trainable, TileLang block ops. "step
peak" and "resident" are the torch allocator's allocated-bytes counters (live tensors,
not reserved or device use; see [../guides/logging.md](../guides/logging.md#vram)):

| profile | ms / step | tokens / s | step peak GiB (allocated) | resident GiB (allocated) | estimate GiB |
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

| case | ms / step | peak GiB (alloc) | H2D GB / step |
| --- | --- | --- | --- |
| resident | 245 | 7.59 | — |
| block checkpoint | 355 | 3.44 | — |
| checkpoint + activation offload | 369 | 3.79 | — |
| checkpoint + stream (pinned) | 439 | 4.36 | 2.33 |
| checkpoint + stream (staged) | 676 | 4.71 | 2.33 |

Each VRAM knob in isolation (`scripts/bench/offload/knobs.py`, one process per case,
8 MOSS-sized synthetic blocks = 0.75 B parameters, 4400 tokens, GRAD_ACC = 2):

| knob | V100 32 GB fp16: ms / step | peak GiB (alloc) | H2D / D2H GB per step | RTX 4090 Windows (4 blocks, bf16): ms / step | peak GiB (alloc) |
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
| text-encoder placements (resident / offload / stream / cpu), encode-ahead | works | works |
| inference plans (`apply_inference_plan`) | works | works |
| multi-GPU (DDP, FSDP2, pipeline, threaded) | Linux-only by design (rejected by knob validation) | same |

The XPU column is `tests/memory/offload` on the B50 (activation offload, streaming,
gradient offload, every optimiser alone and in the session).
