# Tuning

Each tuned kernel picks its tile config, and each block op picks its kernel or
`torch.compile`, by measuring the candidates once per card and storing the winner in a
tuning table. Later processes read the table and time nothing.

| question | answer |
| --- | --- |
| which knob | `TUNE` ([TUNE](#tune)): `table` (training default) or `off` (inference default) |
| when a training run times | the prewarm before the first step, then only inside the [tuning windows](#tuning-windows) |
| how long the retune lasts | `RETUNE_STEPS` (default 3): after each of the first N steps; then kernel-vs-compile queueing is sealed (the first sample grid's window still times its own shapes) |
| where the table lives | `<cache>/tuning/<card-sm-cuda-torch>/<tag>/<kernel>.json` ([Table](#select)), cache root `KOHAKUMOSS_KERNEL_CACHE` ([Cache directory](#cache-directory)) |
| how to fill it ahead of time | `kohakumoss-tune` ([Filling the table](#filling-the-table-kohakumoss-tune)) |

## TUNE

The run knob `TUNE` (training, `sample.py`, `MossPipeline(tune=...)`), applied once per
process with `tuning.set_mode(...)`:

| value | timed | table | behaviour |
| --- | --- | --- | --- |
| `table` (training default) | yes | read, write | a stored decision binds; a missing key is timed once and the winner stored. Training: in the prewarm before the first step (below), then only inside the tuning windows. Inference: on the key's first eager call; an owned pipeline measures queued kernel-vs-compile shapes after each generate |
| `off` (inference default) | no | read | a stored decision binds; a missing key takes the planner's first candidate (a kernel-or-compile op: its kernel), never stored |

`RETUNE_STEPS` (training, int >= 0, default 3) is how many first steps of the process
run the between-step kernel-vs-compile retune under `TUNE="table"`.

### LyCORIS kernels

Under both values the LyCORIS kernels follow TUNE through LyCORIS's process switch
`lycoris.kernels.set_tuning` (LyCORIS >= 4.1.2):

- `kohakumoss.adapters.lycoris.tuning.set_lycoris_tuning` sets it and returns the
  previous setting for `restore_lycoris_tuning`.
- On under `table`, off under `off` (LyCORIS then binds its stored tile plan, else its
  first).
- The trainer sets it next to `tuning.set_mode` and restores it when the run ends; under
  `table` the adapter prewarm before the first step does the LyCORIS tiling, under
  `rng_kept` like every other tuning call.

LyCORIS's own tuning leaves the caller's generators where they were. Its cost model
measures the device once per process, the first time it ranks tiles (a disk miss of its
tile table, in the prewarm or in a live adapted forward that meets a new shape), on
random data from a private generator, so neither the prewarm nor a live forward draws on
the run's random stream. `tests/kernels/tuning/test_rng_kept.py` checks a cold LyCORIS
tile tune on the GPU (negative: the probe without its private generator).

### Decision log

Decisions are logged per problem key through `tuning.decisions`
([logging.md](../guides/logging.md)):

| source | recorded as |
| --- | --- |
| `select`, a stored row that is still a candidate | `table` |
| `select`, timed | `measured` (with the shortlist timings) |
| `select`, `TUNE="off"` with no stored row, or a single candidate | `default` |
| `select`, a provisional first-candidate return (timing not allowed here, a trace) | nothing |
| kernel-or-compile op, a stored row | `table` |
| kernel-or-compile op, `TUNE="off"` without a row | `default` |
| kernel-or-compile op, a problem queued for `retune_pending()` | `queued` (DEBUG); the retune's measurement is the decision |

Each `(kernel, key)` is recorded once per process, so a second call pattern of a decided
problem adds nothing. Batches are flushed as one INFO table: after a prewarm, after each
retune that measured, at the end of training step 1, after a generate. After step 1
(`go_live`) a measured decision is one INFO line naming the key, the choice and the
timings.

### Where nothing is timed

- Inside `torch.compile` tracing or CUDA-graph capture a missing key takes the first
  candidate untimed (not stored, not settled), so a compiled graph carries no tuning
  code.
- `select` times nothing inside an autograd backward (unless the kernel launches no
  autograd, `in_backward=True`).
- For autograd launches (`autograd=True`), nothing under inference mode or saved-tensor
  hooks (a checkpointed forward).

## Tuning windows

A training run tunes kernels only in these windows:

| window | what is timed | memory returned to the driver |
| --- | --- | --- |
| the prewarm before the first step (and `kohakumoss-tune`) | every planned problem | before and after each phase, before each bounded config |
| the online text encoder's prewarm, right before its first pass (`TextPrewarm`) | its pack buckets | after it |
| after each of the first `RETUNE_STEPS` steps of the process (default 3; `RetuneKernels`) | queued kernel-vs-compile problems (`torch.compile` against the kernel) | only when a queued problem's inputs do not fit |
| the first sample grid of the run, up to the end of its first sampler step (`PipelineImages`) | missing configs of the sampling shapes | the sampler's own releases (after encode, after sampling, after decode) |

The window state lives in `kohakumoss_kernels.tuning.window`:

| call | when | effect |
| --- | --- | --- |
| `window.freeze()` | the end of the prewarm (`end_prewarm`) | outside a window `select` times nothing; a key first met there binds the first candidate that builds and runs once, untimed, unstored, named in `tuner.DEFERRED` and logged once at WARNING (`met outside a tuning window`): the prewarm's plan did not reach that shape, and the next run's prewarm covers it once the plan does |
| `window.tuning_window()` | the text-encoder prewarm, the first sampler step | lifts the freeze inside |
| `window.timing_here()` | the retune | lifts the freeze (and the go-live deferral) inside |
| `window.seal()` | after the last retune step (`RetuneKernels`) | a kernel-or-compile op queues no new problem (it binds the kernel); the problems still queued are dropped and stay on the kernel (one INFO line `kernel-vs-compile retune closed after step N`); the callback is a no-op from then on |
| `window.reopen()` | the run's end, with `tuner.clear_caps()` | not frozen, not sealed, not deferring, no process tuning budget: a later inference or tuning call in the same process tunes as a fresh one (the decision log's live mode stays) |

- The retune count is per process, so a resumed run retunes its own first steps.
- Under a process group a window after step 1 lifts the freeze, not the go-live
  deferral, so its missing keys stay untimed there ([Multi-process](#multi-process)).
- The VAE decode of the first sample grid runs after its first sampler step and binds
  untimed where a key is missing.

`tests/kernels/tuning/test_tuning_window.py` checks that a frozen key binds untimed and a
window times it (negative: a select that ignores the freeze), that the retune runs over
the first N steps only and then seals (negative: a retune every step), that a sealed op
queues nothing (negative: queued after the seal), and that only the first sample grid's
first sampler step is open (negative: every grid open).

## Filling the table: kohakumoss-tune

`kohakumoss-tune` (also `kogine run scripts/bench/tune/tune.py`) runs the training
prewarm once for one card, dtype and image-size list, and stores every decision; later
runs bind them without timing.

```
kohakumoss-tune --device cuda:0 --dtype bf16 --sizes 1024x1024 832x1216 --batch 4
kohakumoss-tune --dtype fp16 --sizes 1024x1024 --backward   # training shapes
```

| flag | meaning |
| --- | --- |
| `--device`, `--dtype` | the card and the 16-bit dtype (`bf16` / `fp16`) |
| `--sizes HxW ...` | image sizes; the longest joint sample (latent tokens + `--text-tokens`) sets the per-sample length |
| `--batch` | images per generate batch; the packed length is `sample x batch`, doubled for CFG unless `--backward` |
| `--variant` | the architecture (`base`, `sft`) |
| `--text-tokens` | text tokens per sample added to each size's latent tokens (default 512) |
| `--backward` | forward + backward (training shapes, no CFG doubling) |
| `--trainable` | with `--backward`: `adapter` (default; frozen base weights, activations carry gradients) or `full` (every weight trains) |

The recorded pass runs under the autocast of the run it tunes for, so the stored keys
are the ones that run looks up (`tests/cli/test_tune_autocast.py`, negative: no
autocast):

- **With `--backward`:** the trainer's autocast to the chosen dtype
  (`kohakumoss.precision.autocast_factory`), so the conditioning linears return the
  16-bit shift / scale / gate a training step passes.
- **Without it:** the sampler's autocast (`cli.tune.record_autocast`:
  `precision.autocast_for` the compute dtype, as `inference.pipeline.sampling_autocast`),
  so the stored keys carry the dtypes a sampling call meets.

It logs the problems measured and the table directory, and releases the tuning
workspaces after.

## Select

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
   best of ten calls, host launch cost included as the eager training loop pays it.
   `timing="burst"` (the adapter ops' `AutotunedOp`) times a burst of back-to-back calls
   issued while the device spins (`torch.cuda._sleep` over twice the burst's measured
   host launch time), so every call is queued before the first starts and the burst's
   device time per call is what is compared: a GPU-bound training step hides the host
   launch time of an adapter op, which a single synchronised call would count.
   `timing="graph"` (CUDA-graph replay, launch cost excluded) is for benches. The tuner
   never captures a graph: a capture in the training process makes CUDA calls of other
   threads (the DataLoader's pin-memory thread) fail, in global and, as measured on
   Windows, thread-local capture mode alike. Every config runs under the
   [working-memory bound](#working-memory-bound).
3. **Table.** The winner, every timing, every config's working memory (`working_mib`)
   and every failure are written to
   `<cache>/tuning/<card-sm-cuda-torch>/<tag>/<kernel>.json` (atomic replace), where
   `<tag>` hashes the kohakumoss / lycoris-lora / triton / tilelang versions (a LyCORIS
   release changes the adapter ops' kernel candidates). Later processes read the
   table and never time again; a stored config that is not among the current candidates
   is re-tuned.

A config that fails during timing:

| failure | handling |
| --- | --- |
| a build / launch error (`errors.BUILD_ERRORS`) | recorded and skipped |
| over the working-memory bound | recorded and skipped |
| out of device memory otherwise | skipped and the cache released; the problem binds the best config that ran (else the first) for this process only, nothing stored |
| every config fails | `ConfigsFailed` hands the op to the registry fallback ([kernel-dev.md](kernel-dev.md#fallback)) |

Shortlist per kernel:

| kernel | shortlist |
| --- | --- |
| Triton attention (fwd, dK/dV, dQ, fused bwd) | 10 |
| TileLang attention (fwd, dK/dV, dQ; each config is an nvcc compile) | 6 |
| Triton attention backward strategy (fused vs split) | both |
| attention entry (backend kernels vs torch `varlen_attn` / SDPA loop) | every implementation |
| fused linear, TileLang block ops | every legal config |
| quant matmul (Triton, TileLang) | every tile plus the PyTorch paths |
| kernel vs compile of a block op | both |
| anything else (`select` default) | 4 |

Table writes:

- Each write first merges the rows already in the file, so processes tuning cards of the
  same model at the same time (a bench matrix pool, DDP ranks) keep each other's rows.
- The write goes through a temporary file unique per process, thread and write, replaced
  onto the table (retried while another process holds the file open, a
  `PermissionError` on Windows); threads sharing one table write one at a time.
- A write that still fails (disk full, a read-only cache directory) logs one WARNING per
  file and keeps the rows in memory: the choice binds for this process and is never
  recorded as a backend failure (`hardware.device.save_json`; the device probe, the
  toolchain cache and the failure records write the same way).

### Problem keys

Problem keys bucket the variable dimensions, so a training run with varying resolutions
settles into a handful of keys:

- Token counts, segment counts and the longest segment enter every stored key through a
  power-of-two bucket. Fixed model dimensions, dtypes and flags enter as they are.
- No per-process value (device index, ids, pointers) is part of a stored key, so a
  restart with the same configuration hits the stored rows.
- Attention keys are `<dtype>-h<heads>-d<head dim>-L<longest segment rounded up to a
  power of two, at least 64>-S<segment count bucket: 1, 2, 4, 8 and above>`; the token
  count is not part of the key (the grid covers it).
- The attention entry's implementation choice adds `-g1` (a backward runs, timed
  forward + backward) or `-g0` (forward only, timed forward only).

### Working-memory bound

The VRAM a run needs is not known when it tunes, so tuning keeps its own peak at what
the run's own calls need.

- **Working memory.** A config's working memory is the peak bytes allocated above the
  bytes allocated when its timing starts (`hardware.memory.working`).
- **Bound.** The configs of one problem are timed in shortlist order; each runs with a
  bound of `WORKING_FACTOR` (8) x the leanest working memory of the configs timed before
  it, at least `WORKING_FLOOR` (256 MiB); the first config has no bound.
- **Enforcement.** The allocator enforces the bound: for the timed call, the per-process
  memory fraction (`set_per_process_memory_fraction`, CUDA and XPU) is set to the bytes
  reserved at the start plus the bound, then restored, so the config grows the reserved
  memory by at most the bound; an allocation past it raises out-of-memory before the
  memory is taken.
- **Trim.** In the prewarm every bounded config starts from an empty allocator cache
  (`hardware.memory.trim` before it), so whether a config fits never depends on the free
  blocks earlier configs left cached, and two runs on one card store the same verdict
  (`tests/kernels/tuning/test_tuner_working.py`, negative: no trim).
- **Verdict.** A config over the bound is recorded under `failures` ("over the
  working-memory bound") and the fastest config within the bound is stored as usual (the
  same problem excludes it again in any later process). Where the bound is not below what
  the device has free, or the allocator takes no fraction, the bound is not enforced and
  a real out-of-memory is the per-process skip above.

The rule is not op-specific. Its main case is a PyTorch-path candidate that pads to a
dense layout. The eager `gated_delta_rule_varlen` pads every segment to the longest in
fp32, so on a pack of one 4110-token row and 31 one-token rows it works on 32 x 4160
padded rows:

| case | working memory |
| --- | --- |
| padded path, 2 value heads of 128 dims (host peak, bf16 inputs) | 1040 MiB |
| padded path, the text encoder's 32 value heads | about 16 GiB |
| the same longest row alone, 2 heads | 51 MiB |
| 32 rows of 128 tokens, 2 heads | 83 MiB |

The Triton designs work on the 4141 packed rows: at 32 value heads the chunk design
allocates the output (32 MiB), the chunk states (96 MiB), the T matrix (24 MiB) and the
normalised values (32 MiB). With the bound (8 x the chunk leader's working memory) the
padded path is skipped on the skewed pack and still timed on the others, and the tuning
peak is that of the kernels the run uses.

Candidate lists rank their leading design first (`triton/textenc/delta.candidates`: the
chunk leader, then the PyTorch path), so the first, unbounded config is a lean one.
Where the default is not the lean one, `select(..., lean=i)` times shortlist entry `i`
first and keeps the first candidate as the default: the adapter ops time their body
candidate before the LyCORIS kernel (whose first call also runs LyCORIS's own tile
tuning), see [Adapter autotuned-op prewarm](#adapter-autotuned-op-prewarm).

## Device description

`hardware.device.measure_rates(device)` returns the card's driver limits plus three
measured rates (16-bit GEMM TFLOP/s with fp32 accumulation, DRAM and L2 copy GB/s),
measured once per card / CUDA / torch combination and cached in `<cache>/devices/`.
Inside a trace or graph capture nothing is measured: planners use the unmeasured `query`
description with default rates.

## Kernel vs compile

Every block op and fused linear kernel is registered as a `KernelOrCompile`
(`backends/compile/choice.py`, table `choice_<backend>_<op>`): per problem (bucketed
shapes plus the grad flags of the tensors, `shape_key`) it binds its backend's kernel or
the `compile` backend's callable. Only the most-prior backend's kernel and `compile` are
timed; other backends are reached only through the registry fallback, never built or
timed otherwise.

A call looks its problem up, in this order:

1. a bound callable, per call key: device, grad mode, intra-op threads, autocast dtype
   and every argument's layout (its leading token dim in the `shape_key` bucket) or
   value, so the binding table holds one entry per token bucket however the token count
   varies (`tests/kernels/tuning/test_choice_binding.py`, negative: exact shapes);
2. the settled choice or stored row of an earlier measurement;
3. else the kernel, and the problem is queued (until `window.seal()` stops
   kernel-vs-compile queueing).

A call traced by torch.compile runs the kernel and binds nothing. Nothing is timed in
the call path, so a forward and its checkpoint recompute always bind the same
implementation.

- **`retune_pending()`** runs outside any autograd graph: in training after each of the
  first `RETUNE_STEPS` steps ([Tuning windows](#tuning-windows)), in an owned pipeline
  after each generate. It measures the queued problems on synthetic inputs built in
  their own dtype (forward + backward over the arguments that need grad) and stores the
  winners; each problem's next call binds its winner. A queued problem whose synthetic
  inputs do not fit in device memory is dropped with a WARNING and stays on the kernel.
- **`decide(*args)`** measures one problem directly (benches, prewarm).
- Under `TUNE="off"` neither measures.
- On first use per device the kernel runs once without autograd (`_probe`), so a kernel
  that cannot build hands the op to the registry fallback before any graph is recorded.

An op served by the `compile` backend takes inductor's launch config, which inductor
benchmarks per process (several reduction splits for one generated kernel): two
processes can round its output differently, so cross-run comparisons use the error
statistics, never bitwise equality.

### Pipelines

A `MossPipeline` that owns its model (`from_pretrained`, or the constructor with
`owned=True`) applies its `tune` (default `off`) as the process's TUNE and LyCORIS's:

- `tune="off"`: its calls bind the stored decisions of this card, else the planner's
  first candidate.
- `tune="table"`: a missing key is timed on its first eager call, and the queued
  kernel-vs-compile problems are measured after each generate (`retune_pending`); the
  winners are stored. A pipeline over the shared training model (`from_components`) leaves TUNE to
the trainer.

## Prewarm

The model prewarm (`kohakumoss.tuning.prewarm.ModelPrewarm`) discovers the tuned ops
from the model's structure. The trainer (`training.build.kernels`) and `kohakumoss-tune`
(`kohakumoss.cli.tune`) use the same class.

| part | module |
| --- | --- |
| the prewarm | `kohakumoss.tuning.prewarm.ModelPrewarm` |
| the planned lengths | `kohakumoss.tuning.lengths` |
| the trainer's build | `training.build.kernels` |
| the step callbacks | `training.callbacks.kernels` |

Kernel tuning needs the tensor shapes and dtypes of the calls, not the weights, so it
records from a weightless twin and keeps its own VRAM at what one op at the run's
largest planned size needs. It runs in five stages.

### 1. Twin

`kohakumoss.training.build.shapes.run_shapes` builds the backbone the run builds on the
meta device (`model.build.build_shapes`):

- It applies the run's base storage there with the same functions the memory plan uses
  (`kohakumoss.memory.apply.quantize_base`: int8 / fp8 / fp8_compute / mxfp8 bases with
  their fp8 input-gradient copy; `trainable_compute`: the fp8 compute caches of a
  trainable base).
- It materializes every tensor as a stride-0 view of one zero on the device
  (`model.build.materialize`): one element of memory per tensor, so the twin holds no
  weight memory and the real weights are never read.
- Every parameter takes the dtype and grad flag of the built backbone's parameter of the
  same name and shape (masters, KEEP_FP32 modules, trained norms); norm eps, residual
  scales and the fp16-fix stream config are copied from it; a linear the built backbone
  carries an adapter on returns an output that requires grad, as the adapter's output
  does.
- Quantized bases and fp8 caches are bound to the device's kernels.

`kohakumoss-tune` uses the same `build_shapes` without a built backbone.

### 2. Record

One twin forward + loss on a small synthetic batch runs inside
`kohakumoss_kernels.recording.recording()`, under the run's autocast, with saved-tensor
hooks that keep nothing (`kohakumoss.tuning.prewarm.keep_nothing`: the recorded pass never runs backward,
so it holds no activation or dequantized weight).

- Every `Bound` call is recorded with the autocast state it ran under and answered by
  the eager reference, so nothing is built or tuned.
- Each tensor argument is snapshotted at the call (`Snap`): shape, dtype, device and
  grad flag, plus a device copy of its values when it has at most 65536 elements (ids,
  bounds, positions, per-sample vectors). The recording holds no live tensor.

### 3. Replay

`kohakumoss_kernels.tuning.replay.replay` runs each unique recorded call at every
planned joint-length bucket, one op at a time, forward + backward where an argument
needs grad, under the recorded autocast state, so the replayed problems carry the dtypes
the live step passes.

- **Lengths.** A bucket is replayed at its top or at the planned maximum
  (`planned_lengths`: the largest joint batch the data plan builds), whichever is
  smaller (`block_lengths`); the key of the top bucket is unchanged.
- **Arguments.** They are materialized from the snapshots on the recorded device: kept
  values are resampled along a token dim (`cu_seqlens`-like bounds and the
  longest-segment argument scaled), larger tensors are fresh values of the same layout
  (normal floats; integers uniform over the recorded range). Timing does not depend on
  the values.
- **Memory.** Each problem's tensors are dropped when it ends, and each op's free blocks
  go back to the driver after its problems.
- **Fallback.** Each problem runs through the op's `Bound` as build time
  (`registry.exercise`): a backend that cannot build or launch, in the forward or the
  backward, falls the twin's op and the model's op of the same query back before step 1
  (one WARNING), and a build / toolchain cause is recorded on disk
  ([kernel-dev.md](kernel-dev.md#fallback)).
- **Timing.** Kernel tuning and the kernel-vs-compile measurement run here, every timed
  config under the working-memory bound; the queued kernel-vs-compile problems are
  decided at the end.

**Attention segment buckets.** Attention is warmed at every segment-length bucket up to
the longest planned sample and, per length, at every segment-count bucket (1, 2, 4, then
8 and above) the planned batches reach there, and no other:

- `planned_segments` gives the shortest joint sample and the most samples a batch holds.
- The shortest joint sample is the smallest grid `Geometry.fit` yields at TARGET_AREA
  over ASPECT_RANGE, plus the dataset's shortest caption estimate
  (`MossDataset.text_tokens_min`: the estimate columns the token budget packs by, over
  the live (not tombstoned) rows, and the dropped caption's estimate (the empty prompt)
  whenever caption dropout is on, since a dropped sample carries that text). The
  trainer builds the data loader before the prewarm for it.
- The most samples a batch holds is BATCH_SIZE under count batching, else MAX_TOKENS
  over the shortest sample, capped by MAX_SAMPLES. A batch whose longest sample has
  length L holds at most `1 + (MAX_TOKENS - L) // shortest` of them
  (`reachable_segments`).
- Each call packs one segment of L and the rest of the shortest length, within
  MAX_TOKENS.
- With a dataset the smallest image grid is the dataset's own
  (`MossDataset.image_tokens_min`: the smallest of the resolution planner's per-row
  targets over the live rows; `tests/training/prewarm/test_prewarm_buckets.py`,
  negatives: every row, no dropped caption), so with UPSCALE="never", where an image
  smaller than TARGET_AREA keeps its size, the shortest sample and the segment counts
  follow the data.
- Where no dataset exists the text minimum is the caption-free template
  (`text_overhead_tokens`): `kohakumoss-tune` takes the count from `--batch` (doubled for
  CFG on generate shapes) and the shortest sample from the smallest `--sizes`.
  `ModelPrewarm` called without `segments` / `shortest` takes `batch_max // sample_max`
  samples of the longest length.

Example: MAX_TOKENS 20000 at 1 MP (smallest grid 3942 tokens) with captions of 600
tokens at least reaches 4 samples, so the 8-segment bucket is never warmed.

**Attention streams.** The attention prewarm covers each attention call site of the
recorded pass as one stream (`AttentionStream`): the joint attention (one segment up to
the longest planned sample, the rest the shortest sample) and the text refiner's
(captions from the dataset's shortest to its longest estimate, else MAX_TEXT_TOKENS).
Each is forward + backward where the recorded call takes a gradient and forward only
(`-g0` keys) where it does not, as the frozen refiner of an adapter run.

A key first met in a live call after the prewarm binds untimed outside the tuning
windows ([Tuning windows](#tuning-windows)). The adapter token buckets are capped the
same way (`token_buckets`). The VAE has no tuned op.

### 4. Decided problems are not run

Each replayed problem (an op call at one planned bucket; attention: its bucket sweep)
has a manifest key: a hash of the op, its bound backend, the autocast state, the
token-dim map and the argument layouts (shapes, dtypes, grad flags, kept integer
contents).

- While it runs, `tuner.watching()` collects the `(kernel, key)` of every tuned choice it
  looks up (`select` with more than one candidate, the kernel-vs-compile table lookups).
- After the kernel-vs-compile decisions, every problem whose choices are all stored is
  written to the `prewarm_problems` table.
- A later prewarm on the same card and package versions does not run a problem whose
  manifest row lists only stored choices; its live calls bind from the tables.
- A problem whose row is missing, or one of whose choices is not stored, runs again.
- The between-step `retune_pending()` only measures problems a live call queued because
  their table row was missing, so a decided key is never timed again.

### 5. Report

One table per device: `op | from table | tuned | skipped | timed | secs | peak GiB |
reserved GiB | bound`, with a total row, then one line `prewarm: N problems from the
tuning table, M tuned in X s`.

| column | meaning |
| --- | --- |
| `from table` | the problems already decided (not run) |
| `timed` | the shortlists timed |
| `peak GiB`, `reserved GiB` | each op's device peak during its problems |
| `bound` | the implementations a kernel-or-compile op picked (`kernel` / `compile`), or the bound backend of any other op |
| `skipped` | a replayed problem that still raises after the fallback (no backend left, an error that is not a backend failure, out-of-memory): logged as a warning and left to the live step |

At the end of step 1 (`PrewarmVram`) one line sets the run's prewarm peak against step
1's. The peaks start empty at each run's `build_prewarm`, so a process that trains
several runs reports each run's own. Step 1's peak is one window over the whole step
(reopened after a prewarm inside it), so the text-encoder and VAE phase marks inside
the step cannot cut it.

### Text-encoder prewarm

The text-encoder ops are tuned only when the run encodes text in the loop
(`TEXT_CACHE="online"` or `"off"`), and only when it does. `attach_text_prewarm` hands
the encoder a `TextPrewarm` that runs once right before its first pass
(`QwenTextEncoder.before_first`), so a run whose captions all hit the online cache never
tunes them. It runs the text prewarm ([Text-encoder buckets](#text-encoder-buckets)),
logs its own prewarm table, keeps its peak for the step-1 line and returns its
workspaces to the driver; the step-1 VRAM window and the step-1 speed leave it out
(`late_prewarm`).

### Prewarm tests

| test | checks | negative |
| --- | --- | --- |
| `tests/training/prewarm/test_replay_stream.py` | the snapshot layouts | |
| `tests/kernels/tuning/test_replay.py` | the recorded autocast state; a failing replay rebinds the recorded op and the model's op | a replay on the implementation |
| `tests/training/prewarm/test_prewarm_shapes.py` | the twin of an int8-base run holds one element per tensor with the int8 bases of every quantized linear, the built backbone's dtypes and grad flags; the recorded pass keeps no saved tensor | a weight-sized module, a fp32 base, plain autograd |
| `tests/training/prewarm/test_prewarm_buckets.py` | the replay points and the attention calls stay within the planned maximum while covering every bucket | the bucket tops |
| `tests/kernels/tuning/test_tuner_working.py` | the working-memory bound | no bound |

### Adapter autotuned-op prewarm

The adapter prewarm (`AdapterSet.prewarm`) settles each distinct `AutotunedOp` once
([adapter-binding.md](adapter-binding.md#autotuned-ops)).

- Ops whose `signature` matches (the tuning name, i.e. op and backend, and the layouts
  of every operand without a token axis) share their problems; the problem key is a
  function of exactly these, the token bucket and the grad pattern.
- Within an op, a bucket and grad pattern already settled in process or stored in this
  card's table (`lycoris_<op>_<backend>.json`) is bound without timing, so a restart
  against the same kernel cache times nothing.
- The prewarm logs the autotuned ops, the distinct ones, the problems measured and its
  device peak.

**Memory.** It follows the kernel prewarm's rules:

- The scratch inputs are constructed at the token buckets capped at the planned maximum
  (`token_buckets`).
- Problems run one at a time: each runtime input is built when its problem is reached
  (`runtime.problems(rows)`: a builder per bucket and grad pattern) and dropped after
  it, and a measured problem's free blocks go back to the driver (`release`) before the
  next, so the cached blocks of one token count never stack under the next.
- Each problem's body candidate is timed first (`select(lean=BODY)`).
- Before a problem is timed, the LyCORIS kernel runs once on copies of its operands
  (forward, and backward under grad), untimed, within the tuning budget below but
  outside the working-memory bound (`AutotunedOp._settle_tiles`). That call holds LyCORIS's cold tile tuning, whose
  inputs, candidate outputs and split-K partials are several times the kernel's own
  working memory: 470-880 MiB at 1024-8192 tokens for the LoKr factor-8 linears of the
  model. Inside the timed call, that tuning passed the bound of 8 x the body's working
  memory (at least 256 MiB). The kernel was then skipped as "over the working-memory
  bound", and every larger bucket inherited the body. With the tiles settled first, the
  bound and the timing see the kernel's own calls.
- The prewarm starts and ends on an empty allocator cache and its device peak enters
  `note_peak`, so the step-1 line `VRAM peak: prewarm` and the `prewarm_peak_*` metrics
  cover it.

**Budget.** Every candidate's timing, LyCORIS's own cold tile tuning inside the kernel's
first call included, runs under an absolute budget, `hardware.memory.tuning_budget`: the
kernel prewarm's device peak (the run's block ops at the planned maximum, below the
training steps' peak) minus the bytes allocated at that point, at most
`BUDGET_FRACTION` (0.5) of the device's free memory after the run's resident state.

- **Cached runs.** A cached kernel prewarm replays little or nothing, so its ops' peaks
  sit near the resident baseline and would give a budget near zero. The kernel prewarm
  therefore stores its working bytes (its device peak minus the bytes allocated when it
  started) in the prewarm manifest under one key per prewarm
  (`prewarm_working|<hash of its replay problem keys and attention key>`).
  `note_prewarm` keeps as its peak the peaks of the ops that ran together with the
  current baseline plus the stored working bytes, storing the larger working back; a
  cached op's near-baseline peak never counts. A fully cached run thus gets the budget
  of the run that measured it. With nothing replayed and nothing stored (a manifest
  older than the stored value) no kernel-prewarm peak is noted and the free-memory term
  applies alone.
- **Several processes.** Only rank 0 runs the kernel prewarm, while every rank runs the
  adapter prewarm and times its own candidates (`training/entry.py` `_prewarm`). Rank 0
  releases the prewarm gate with its working bytes as the gate's payload
  (`share_working` -> `release_ranks(gate, payload=...)`, the store key the other ranks
  already wait on, no extra collective), and each other rank keeps its currently
  allocated bytes plus those working bytes as its prewarm peak (`take_working`), so
  every rank's adapter budget is rank 0's.
- **Over the budget.** A token bucket over the budget is never timed and inherits
  instead (`tuner.inherit`): the decision of the largest measured bucket of the same op
  and grad pattern, stored with `"inherited": <source key>` and logged under
  `inherited`, so a live call at that size binds it and never tunes. A bucket is over
  the budget when the largest working memory of its source bucket, scaled by the token
  rows, passes it, or when a candidate's timing reaches the cap (`select(cap=...)`
  raises `OverBound` before the allocation, nothing stored). The smallest bucket of a
  pattern is always timed. The kernel stays the default where nothing is timed, and the
  relative bound (8 x the leanest) applies as well.

**The cap after the prewarm.** The end of the adapter prewarm (`prewarm.end_prewarm`)
makes that budget the process's:

- `tuner.set_cap(device, budget)`, so every later `select` that passes no cap (inside
  the tuning windows: the between-step kernel-vs-compile retune, the text-encoder
  prewarm in step 1, the first sampler step) times within it; a key whose timing would
  pass it binds its first candidate for the process, untimed and unstored
  (`tuner.DEFERRED`, one INFO line `over the tuning budget`).
- A tune after the prewarm leaves its working memory in the allocator's cache for the
  steps to reuse; a config that runs out of memory returns the free blocks (`release`).
- It turns LyCORIS's tile tuning off (`set_lycoris_tuning(False)`, restored at the run's
  end): a LyCORIS key the prewarm did not tune (an inherited bucket's size) binds
  LyCORIS's stored or model-ranked tile and is never tuned inside a step.
- The adapter prewarm flushes its decisions as their own table (`after the adapter
  prewarm`), so the first retune's table lists only the retune's decisions.

`tests/kernels/tuning/test_tuner_working.py::test_live_tunes_stay_within_the_budget`
(negative: no cap) and
`tests/training/prewarm/test_adapter_prewarm.py::test_prewarm_end_caps_every_later_tune`
(negative: LyCORIS left tuning) check it.

**Measured case.** The kernel candidate is the heavy one where the input takes a
gradient. On an RTX 5090 (LoKr full_matrix factor 16, fp16, the 3072 -> 16384 bypass with
base):

| rows | kernel working MiB | compiled body working MiB |
| --- | --- | --- |
| 20000 | 6680 | 2500 |
| 16384 | 5472 | 2048 |
| 8192 | 3040 | 1024 |

The kernel is the faster one from 4096 rows (5.6 against 9.1 ms at 20000). Held in the
allocator cache from one problem to the next, those blocks stacked to 12.8 GB in
nvidia-smi (11.5 GB on LyCORIS 4.1.2) against 6.6 GB for the training steps and a 4.0 GiB
kernel prewarm peak; with that peak as the budget, the 16384- and 20000-row buckets
inherit the 8192-row decision (the kernel).

`tests/training/prewarm/test_adapter_prewarm.py` checks:

- only one problem's inputs are alive while it is timed (negative: every bucket's inputs
  built up front);
- an over-budget bucket inherits the largest fitting bucket's decision without being
  timed, by prediction and by the cap (negative: no budget, the bucket is timed);
- a fully cached kernel prewarm gives the budget of the uncached run and a never-measured
  key the free-memory term (negative: the stored working bytes ignored);
- the reported peak holds the adapter prewarm (negative: the peak not noted).

### Text-encoder buckets

The packed text-encoder ops key their problems on buckets of the pack shape, defined
once in `kohakumoss_kernels.tuning.buckets` and shared by the kernels and the prewarm:

| op | key dims |
| --- | --- |
| `gated_delta_rule_varlen` (triton, tilelang) | segments: 1, 2, 4, then 8 and above as one; longest segment: pow2 >= 64; mean segment: pow2 >= 64 |
| `attention_varlen_causal` (triton, tilelang) | longest segment: pow2 >= 64 (the segment count is not a key dim) |
| kernel-or-compile `rms_norm` (q / k / layer norms) | tokens: pow2 >= 256 (`shape_key`) |

Runtime integers (tokens, segment count, longest segment) are kernel arguments, not
compile-time constants. TileLang compiles per head dims, tile config and dtype only;
Triton specializes integer arguments only on divisibility by 16 and equality to 1. A new
value inside a bucket therefore reuses the compiled kernel. The keys carry a version tag
(`KEY_VERSION`), so rows stored under an older key format are never matched.

`kohakumoss.encoders.text.prewarm.prewarm_text` records the kernel calls of one
synthetic two-row pack through a weightless twin of the encoder's text model
(`text_shapes`: the model class built on the meta device from the encoder's config, the
plan's quantized linears, each tensor in the dtype of the encoder's tensor of the same
name, materialized as stride-0 views, quantized GEMMs bound to the device), run on the
encoder's own ops under no_grad; the placed weights are never read.

It replays the calls at one target pack per bucket the plan can produce:

- packs of at most `plan.batch` rows and `plan.max_tokens` tokens;
- rows up to `min(MAX_TEXT_TOKENS + chat template, encoder max length)` and, when the
  trainer knows the data, at most the length bucket of the longest caption estimate plus
  the template (`longest_row`; `MossDataset.text_tokens_max`, the planner's estimate
  columns; the bucket rounding covers an estimate a little below the tokenized length,
  and a longer caption is tuned on its first encode);
- `pack_lengths` builds the row lengths: for every (segments, longest, mean) bucket one
  long row plus equal shorter rows whose mean lands in the bucket, then packs for every
  token bucket not yet reached.

The skewed packs (one long row and many one-token rows) are where the padded PyTorch
delta-rule path is at its largest; the working-memory bound keeps it from setting the
tuning peak. `tests/inference/test_text_prewarm.py` checks on CPU that the planner covers
the buckets of random `plan_groups` packs (negative: equal rows only) and that the data
cap bounds the rows (negative: no cap).

## Multi-process

**Broadcast.** `tuning.broadcast(src=0)` replaces every rank's tables with rank `src`'s
(memory and disk) over the initialised process group, then runs `tuner.reset()`: settled
choices and every binding cache registered with `tuner.on_reset` (kernel-or-compile
block ops, fused-linear runners, attention, text-encoder, optimizer and quant bindings)
are dropped, so live calls bind from the shared tables.

- The trainer prewarms on rank 0 only, broadcasts, then starts; with identical cards
  every rank runs the same configs (a rank on a different card has no rows for its card
  and tunes its own).
- The text-encoder prewarm runs on each rank before that rank's first encode, without a
  broadcast (a rank whose captions are all cached never encodes); each rank's text
  encoder encodes only its own prompts.
- Problems timed after the prewarm (inside the tuning windows; the adapter and
  text-encoder prewarms run on every rank) are timed by the rank that meets them.

**Agree.** After each retune step, on every rank, `RetuneKernels` runs `tuner.agree()`:

- each rank's measurements since the last step are counted with one one-element
  all-reduce (`positive_anywhere`, on the backend's device) and gathered
  (`all_gather_object`) only when some rank measured something, so a step after warm-up
  costs one small all-reduce;
- the lowest rank's measurement of each (card, kernel, key) wins (`agreed`), and every
  rank with a device of that card settles it, stores it, and drops its bindings when the
  winner replaces a different local choice (`adopt`), so every rank binds the same
  config from the next step on;
- one process (or a single-rank group) skips the exchange.

**Deferral.** Inside a step under a process group of two or more ranks no key is timed
(its peers would otherwise wait in the step's collectives under the NCCL timeout):

- From `decisions.go_live` (the end of step 1) on, `tuner.defer_if_grouped` makes
  `select` bind the first candidate of a missing key that builds and runs once, untimed
  (a config raising a build error is skipped as in timing; all failing raises
  `ConfigsFailed`), for the process, unstored, named in `tuner.DEFERRED` and logged once
  at INFO (`deferred under DDP: first candidate, untimed`).
- The between-step retune times inside `window.timing_here()`.
- A sample grid's first sampler step (rank 0 only) stays deferred under a process group:
  its window lifts the freeze, not the deferral.
- The retune times only kernel-vs-compile problems, whose call layouts it queues; a
  deferred generic `select` key (attention, quantized GEMM, adapter configs) keeps no
  call to replay, so under DDP it stays on its first candidate for the run, and the next
  run's prewarm covers it when its plan reaches that shape.

`tests/kernels/tuning/test_tuner_agree.py` checks the deferral, the single-process path
and the retune (negative: no deferral, timed in the step), the merge (negative: the
highest rank wins), the rebinding (negative: bindings kept) and that the objects are
gathered only after a measurement (negative: a gather every step).

## Cache directory

`KOHAKUMOSS_KERNEL_CACHE` moves the cache root (default `~/.cache/kohakumoss_kernels`).
The test session points it at an empty temporary directory before any table is read
(`tests/conftest.py`), so every test runs one fixed config per op and never reads the
host's tables; a test that tunes sets `table` and its own cache directory.
