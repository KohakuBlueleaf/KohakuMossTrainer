# Tuning

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
   best of ten calls, host launch cost included as the eager training loop pays it. The
   tuner never captures a CUDA graph: a capture in the training process makes CUDA
   calls of other threads (the DataLoader's pin-memory thread) fail, in global and, as
   measured on Windows, thread-local capture mode alike. `timing="graph"` (CUDA-graph
   replay, launch cost excluded) is for benches. Every config runs under the
   working-memory bound below. A config that raises a build / launch error
   (`errors.BUILD_ERRORS`) is recorded and skipped; a config that goes over the bound
   is recorded and skipped; a config that runs out of device memory otherwise is
   skipped and the cache released, and the problem binds the best config that ran
   (else the first) for this process only, nothing stored; when every config fails,
   `ConfigsFailed` hands the op to the registry fallback
   ([kernel-dev.md](kernel-dev.md#fallback)). Shortlist per kernel:

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
3. **Table.** The winner, every timing, every config's working memory (`working_mib`)
   and every failure are written to
   `<cache>/tuning/<card-sm-cuda-torch>/<tag>/<kernel>.json` (atomic replace), where
   `<tag>` hashes the kohakumoss / triton / tilelang versions. Each write first merges
   the rows already in the file, so processes tuning cards of the same model at the
   same time (a bench matrix pool, DDP ranks) keep each other's rows. Later processes read
   the table and never time again; a stored config that is no longer among the current
   candidates is re-tuned. The write goes through a temporary file unique per process,
   thread and write, replaced onto the table (retried while another process holds the
   file open, a `PermissionError` on Windows); threads sharing one table write one at a
   time. A write that still fails (disk full, a read-only cache directory) logs one
   WARNING per file and keeps the rows in memory: the choice binds for this process and
   is never recorded as a backend failure (`hardware.device.save_json`; the device
   probe, the toolchain cache and the failure records write the same way).

Problem keys bucket the variable dimensions, so a training run with varying resolutions
settles into a handful of keys. Attention keys are `<dtype>-h<heads>-d<head dim>-L<longest
segment rounded up to a power of two, at least 64>-S<segment count bucket: 1, 2, 4, 8 and
above>`; the token count is not part of the key (the grid covers it). The attention
entry's implementation choice adds `-g1` (a backward runs, timed forward + backward) or
`-g0` (forward only, timed forward only).

### Working-memory bound

The VRAM a run needs is not known when it tunes, so tuning keeps its own peak at what
the run's own calls need. A config's working memory is the peak bytes allocated above
the bytes allocated when its timing starts (`hardware.memory.working`). The configs of
one problem are timed in shortlist order; each runs with a bound of
`WORKING_FACTOR` (8) x the leanest working memory of the configs timed before it, at
least `WORKING_FLOOR` (256 MiB); the first config has no bound. The bound is enforced
by the allocator: for the timed call, the per-process memory fraction
(`set_per_process_memory_fraction`, CUDA and XPU) is set to the bytes reserved at the
start plus the bound, then restored, so the config grows the reserved memory by at
most the bound; an allocation past it raises out-of-memory before the memory is
taken. In the prewarm every bounded config starts from an empty allocator cache
(`hardware.memory.trim` before it), so whether a config fits never depends on the
free blocks earlier configs left cached, and two runs on one card store the same
verdict (`tests/kernels/tuning/test_tuner_working.py`, negative: no trim); a live tune
inside the training loop does not trim (see below). Such a config is recorded under `failures`
("over the working-memory bound") and the fastest config within the bound is stored as
usual (the same problem excludes it again in any later process). Where the bound is
not below what the device has free, or the allocator takes no fraction, the bound is
not enforced and a real out-of-memory is the per-process skip above.

The rule is not op-specific. Its main case is a PyTorch-path candidate that pads to a
dense layout: the eager `gated_delta_rule_varlen` pads every segment to the longest in
fp32, so on a pack of one 4110-token row and 31 one-token rows it works on 32 x 4160
padded rows. Its working memory grows with the value heads: 1040 MiB at 2 value heads
of 128 dims (host peak, bf16 inputs), about 16 GiB at the text encoder's 32, against
51 MiB at 2 heads for the same longest row alone and 83 MiB for 32 rows of 128 tokens.
The Triton designs work on the 4141 packed rows: at 32 value heads the chunk design
allocates the output (32 MiB), the chunk states (96 MiB), the T matrix (24 MiB) and
the normalised values (32 MiB). With the bound (8 x the chunk leader's working
memory) the padded path is skipped on the skewed pack and still timed on the others,
and the tuning peak is that of the kernels the run uses. Candidate lists rank their
leading design first (`triton/textenc/delta.candidates`: the chunk leader, then the PyTorch path),
so the first, unbounded config is a lean one. Where the default is not the lean one,
`select(..., lean=i)` times shortlist entry `i` first and keeps the first candidate as
the default: the adapter ops time their body candidate before the LyCORIS kernel
(whose first call also runs LyCORIS's own tile tuning), see
[Adapter autotuned-op prewarm](#adapter-autotuned-op-prewarm).

## Device description

`hardware.device.measure_rates(device)` returns the card's driver limits plus three measured
rates (16-bit GEMM TFLOP/s with fp32 accumulation, DRAM and L2 copy GB/s), measured
once per card / CUDA / torch combination and cached in `<cache>/devices/`. Inside a
trace or graph capture nothing is measured: planners use the unmeasured `query`
description with default rates.

## TUNE

The run knob `TUNE` (training, `sample.py`, `MossPipeline(tune=...)`), applied once per
process with `tuning.set_mode(...)`:

| value | timed | table | behaviour |
| --- | --- | --- | --- |
| `table` (training default) | yes | read, write | a stored decision binds; a missing key is timed once, on its first eager call, and the winner stored. The trainer (and `kohakumoss-tune`) runs the prewarm below before the first real call and then tunes only inside its tuning windows (below); an owned pipeline measures queued kernel-vs-compile shapes after each generate |
| `off` (inference default) | no | read | a stored decision binds; a missing key takes the planner's first candidate (a kernel-or-compile op: its kernel), never stored; no prewarm |

Under both values the LyCORIS kernels follow TUNE through LyCORIS's process switch
`lycoris.kernels.set_tuning` (LyCORIS >= 4.1.2; `kohakumoss.adapters.lycoris.tuning.
set_lycoris_tuning`, which returns the previous setting for
`restore_lycoris_tuning`): on under `table`, off under `off` (LyCORIS then binds its
stored tile plan, else its first). The trainer sets it next to `tuning.set_mode` and
restores it when the run ends; under `table` the adapter prewarm before step 0 does
the LyCORIS tiling, under `rng_kept` like every other tuning call. LyCORIS's own
tuning leaves the caller's generators where they were: its cost model measures the
device once per process, the first time it ranks tiles (a disk miss of its tile
table, in the prewarm or in a live adapted forward that meets a new shape), on random
data from a private generator, so neither the prewarm nor a live forward draws on the
run's random stream. `tests/kernels/tuning/test_rng_kept.py` checks a cold LyCORIS tile tune
on the GPU (negative: the probe without its private generator).

Decisions are logged per problem key through `tuning.decisions`
(docs/guides/logging.md): `select` records a settled choice as `table` (a stored row
that is still a candidate), `measured` (with the shortlist timings) or `default`
(`TUNE="off"` with no stored row, or a single candidate); a provisional first-candidate
return (timing not allowed here, a trace) records nothing. A kernel-or-compile op
records `table` on a stored row and `default` under `TUNE="off"` without one; a problem
it queues for `retune_pending()`
logs only `queued` (DEBUG), and the retune's measurement is the decision. Each `(kernel, key)` is
recorded once per process, so a second call pattern of a decided problem adds nothing.
Batches are flushed as one INFO table (after a prewarm, after each retune that measured,
at the end of training step 1, after a generate); after step 1 (`go_live`) a measured
decision is one INFO line naming the key, the choice and the timings.

Inside `torch.compile` tracing or CUDA-graph capture a missing key takes the first
candidate untimed (not stored, not settled), so a compiled graph carries no tuning code.
`select` also times nothing inside an autograd backward (unless the kernel launches no
autograd, `in_backward=True`), and for autograd launches (`autograd=True`) nothing under
inference mode or saved-tensor hooks (a checkpointed forward).

## Kernel vs compile

Every block op and fused linear kernel is registered as a `KernelOrCompile`
(`backends/compile/choice.py`, table `choice_<backend>_<op>`): per problem (bucketed
shapes plus the grad flags of the tensors, `shape_key`) it binds its backend's kernel or
the `compile` backend's callable. Only the most-prior backend's kernel and `compile` are
timed; other backends are reached only through the registry fallback, never built or
timed otherwise. A call looks its problem up: a bound callable (per call key: device,
grad mode, intra-op threads, autocast dtype and every argument's layout, its leading
token dim in the `shape_key` bucket, or value, so the binding table holds one entry
per token bucket however the token count varies; `tests/kernels/tuning/
test_choice_binding.py`, negative: exact shapes), else
the settled choice or stored row of an earlier measurement, else the kernel, and the
problem is queued. A call traced by torch.compile runs the kernel and binds nothing.
Nothing is timed in the call path, so a forward and its checkpoint recompute always
bind the same implementation. `retune_pending()`, called between steps outside any
autograd graph, measures the queued problems on synthetic inputs built in their own
dtype (forward + backward over the arguments that need grad) and stores the winners;
each problem's next call binds its winner. A queued problem whose synthetic inputs do
not fit in device memory is dropped with a WARNING and stays on the kernel.
`decide(*args)` measures one problem directly (benches, prewarm). Under `TUNE="off"`
neither measures. On first use per device the kernel runs once without autograd
(`_probe`), so a kernel that cannot build hands the op to the registry fallback before
any graph is recorded.

### Pipelines

Inference runs no prewarm. A `MossPipeline` that owns its model (`from_pretrained`, or
the constructor with `owned=True`) applies its `tune` (default `off`) as the process's
TUNE and LyCORIS's: its calls bind the stored decisions of this card, else the defaults.
`tune="table"` additionally times a missing key on its first eager call and measures the
queued kernel-vs-compile problems after each generate (`retune_pending`), storing the
winners. A pipeline over the shared training model (`from_components`) leaves TUNE to
the trainer.

### Filling the table: kohakumoss-tune

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
| `--backward` | forward + backward (training shapes, no CFG doubling) |

With `--backward` the recorded pass runs under the trainer's autocast to the chosen
dtype (`kohakumoss.precision.autocast_factory`), so the conditioning linears return the
16-bit shift / scale / gate a training step passes and the stored keys are the ones the
trainer looks up; without it the pass runs under the sampler's autocast
(`cli.tune.record_autocast`: `precision.autocast_for` the compute dtype, as
`inference.pipeline.sampling_autocast`), and so does the inference prewarm
(`inference.warm.prewarm_autocast`), so the stored keys carry the dtypes a sampling
call meets (`tests/cli/test_tune_autocast.py`, negative: no autocast). It logs
the problems measured and the table directory, and releases the tuning workspaces
after.

## Prewarm

The model prewarm (`kohakumoss.tuning.prewarm.ModelPrewarm`; the planned lengths in
`kohakumoss.tuning.lengths`, the trainer's build in `training.build.kernels`, the step
callbacks in `training.callbacks.kernels`; inference and `kohakumoss-tune` use the same
class) discovers the
tuned ops from the model's structure instead of a hand list. Kernel tuning needs the
tensor shapes and dtypes of the calls, not the weights, so it records from a
weightless twin and keeps its own VRAM at what one op at the run's largest planned
size needs:

1. **Twin.** `kohakumoss.training.build.shapes.run_shapes` builds the backbone the run
   builds on the meta device (`model.build.build_shapes`), applies the run's base
   storage there with the same functions the memory plan uses
   (`kohakumoss.memory.apply.quantize_base`: int8 / fp8 / fp8_compute / mxfp8 bases with
   their fp8 input-gradient copy; `trainable_compute`: the fp8 compute caches of a
   trainable base), then materializes every tensor as a stride-0 view of one zero on
   the device (`model.build.materialize`): one element of memory per tensor, so the
   twin holds no weight memory and the real weights are never read. Every parameter
   takes the dtype and grad flag of the built backbone's parameter of the same name and
   shape (masters, KEEP_FP32 modules, trained norms); norm eps, residual scales and
   the fp16-fix stream config are copied from it; a linear the built backbone carries
   an adapter on returns an output that requires grad, as the adapter's output does.
   Quantized bases and fp8 caches are bound to the device's kernels. `kohakumoss-tune`
   uses the same `build_shapes` without a built backbone.
2. **Record.** One twin forward + loss on a small synthetic batch runs inside
   `kohakumoss_kernels.recording.recording()`, under the run's autocast, with
   saved-tensor hooks that keep nothing (`kernels.keep_nothing`: the recorded pass
   never runs backward, so it holds no activation or dequantized weight). Every `Bound`
   call is recorded with the autocast state it ran under and answered by the eager
   reference, so nothing is built or tuned. Each tensor argument is snapshotted at the
   call (`Snap`): shape, dtype, device and grad flag, plus a device copy of its values
   when it has at most 65536 elements (ids, bounds, positions, per-sample vectors). The
   recording holds no live tensor.
3. **Replay.** `kohakumoss_kernels.tuning.replay.replay` runs each unique recorded call
   at every planned joint-length bucket, one op at a time, forward + backward where an
   argument needs grad, under the recorded autocast state, so the replayed problems
   carry the dtypes the live step passes. A bucket is replayed at its top or at the
   planned maximum (`planned_lengths`: the largest joint batch the data plan builds),
   whichever is smaller (`block_lengths`); the key of the top bucket is unchanged.
   Arguments are materialized from the snapshots on the recorded device: kept values
   are resampled along a token dim (`cu_seqlens`-like bounds and the longest-segment
   argument scaled), larger tensors are fresh values of the same layout (normal floats;
   integers uniform over the recorded range); each problem's tensors are dropped when
   it ends, and each op's free blocks go back to the driver after its problems. Timing
   does not depend on the values. Each problem runs through the op's `Bound` as build
   time (`registry.exercise`): a backend that cannot build or launch, in the forward or
   the backward, falls the twin's op and the model's op of the same query back before
   step 1 (one WARNING), and a build / toolchain cause is recorded on disk
   ([kernel-dev.md](kernel-dev.md#fallback)). Kernel tuning and the kernel-vs-compile measurement run
   here, every timed config under the working-memory bound; the queued
   kernel-vs-compile problems are decided at the end. Attention is warmed at every
   segment-length bucket up to the longest planned sample and, per length, at every
   segment-count bucket (1, 2, 4, then 8 and above) the planned batches reach there,
   and no other: `planned_segments` gives the shortest joint sample (the smallest grid
   `Geometry.fit` yields at TARGET_AREA over ASPECT_RANGE, plus the dataset's
   shortest caption estimate, `MossDataset.text_tokens_min`: the estimate columns the
   token budget packs by, over the live (not tombstoned) rows, and the dropped
   caption's estimate (the empty prompt) whenever caption dropout is on, since a
   dropped sample carries that text; the trainer builds the data loader before the
   prewarm for it) and the most samples a batch holds (BATCH_SIZE under count batching, else
   MAX_TOKENS over the shortest sample, capped by MAX_SAMPLES); a batch whose longest
   sample has length L holds at most `1 + (MAX_TOKENS - L) // shortest` of them
   (`reachable_segments`). Each call packs one segment of L and the rest of the
   shortest length, within MAX_TOKENS. Example: MAX_TOKENS 20000 at 1 MP (smallest
   grid 3942 tokens) with captions of 600 tokens at least reaches 4 samples, so the
   8-segment bucket is never warmed. Where no dataset exists the text minimum is the
   caption-free template (`text_overhead_tokens`): `kohakumoss-tune` takes the count
   from `--batch` (doubled for CFG on generate shapes) and the shortest sample from
   the smallest `--sizes`; inference takes `batch // sample` samples of the longest
   length. With a dataset the smallest image grid is the dataset's own
   (`MossDataset.image_tokens_min`: the smallest of the resolution planner's per-row
   targets over the live rows; `tests/training/prewarm/test_prewarm_buckets.py`,
   negatives: every row, no dropped caption), so with UPSCALE="never", where an image smaller than TARGET_AREA keeps
   its size, the shortest sample and the segment counts follow the data. The attention
   prewarm covers each attention call site of the recorded pass as one stream
   (`AttentionStream`): the joint attention (one segment up to the longest planned
   sample, the rest the shortest sample) and the text refiner's (captions from the
   dataset's shortest to its longest estimate, else MAX_TEXT_TOKENS), each forward +
   backward where the recorded call takes a gradient and forward only (`-g0` keys)
   where it does not, as the frozen refiner of an adapter run. A key still first met
   in a live call is timed there (TUNE="table") and made the same on every rank by the
   between-step `tuner.agree` ([Multi-process](#multi-process)). The adapter token
   buckets are capped the same way (`token_buckets`). The VAE has no tuned op.
4. **Decided problems are not run.** Each replayed problem (an op call at one planned
   bucket; attention: its bucket sweep) has a manifest key: a hash of the op, its
   bound backend, the autocast state, the token-dim map and the argument layouts
   (shapes, dtypes, grad flags, kept integer contents). While it runs,
   `tuner.watching()` collects the `(kernel, key)` of every tuned choice it looks up
   (`select` with more than one candidate, the kernel-vs-compile table lookups). After
   the kernel-vs-compile decisions, every problem whose choices are all stored is written to the
   `prewarm_problems` table. A later prewarm on the same card and package versions
   does not run a problem whose manifest row lists only stored choices; its live
   calls bind from the tables. A problem whose row is missing, or one of whose
   choices is no longer stored, runs again. Between steps, `retune_pending()` only
   measures problems a live call queued because their table row was missing, so a
   decided key is never timed again.
5. **Report.** One table per device: `op | from table | tuned | skipped | timed | secs
   | peak GiB | reserved GiB | bound`, with a total row, then one line
   `prewarm: N problems from the tuning table, M tuned in X s`. `from table` counts the
   problems already decided (not run); `timed` counts the shortlists timed; the peaks
   are each op's device peak during its problems; `bound` lists the implementations a
   kernel-or-compile op picked (`kernel` / `compile`) or the bound backend of any other
   op. A replayed problem that still raises after the fallback (no backend left, an
   error that is not a backend failure, out-of-memory) is logged as a warning, counted
   under `skipped`, and left to the live step. At the end of step 1
   (`PrewarmVram`) one line sets the run's prewarm peak against step 1's (the peaks
   start empty at each run's `build_prewarm`, so a process that trains several runs
   reports each run's own).

The text-encoder ops are tuned only when the run encodes text in the loop
(`TEXT_CACHE="online"` or `"off"`), and only when it does: `attach_text_prewarm` hands the
encoder a `TextPrewarm` that runs once right before its first pass
(`QwenTextEncoder.before_first`), so a run whose captions all hit the online cache
never tunes them. It runs the text prewarm below, logs its own prewarm table, keeps
its peak for the step-1 line and returns its workspaces to the driver; the step-1 VRAM
window and the step-1 speed leave it out (`late_prewarm`).

Problem keys are bucketed: token counts, segment counts and the longest segment
enter every stored key through a power-of-two bucket. Fixed model dimensions, dtypes
and flags enter as they are, and no per-process value (device index, ids, pointers)
is part of a stored key, so a restart with the same configuration hits the stored
rows.

`tests/training/prewarm/test_replay_stream.py` checks the snapshot layouts;
`tests/kernels/tuning/test_replay.py` checks the recorded autocast state and that a failing
replay rebinds the recorded op and the model's op (negative: a replay on the
implementation). `tests/training/prewarm/test_prewarm_shapes.py` checks that the
twin of an int8-base run holds one element per tensor with the int8 bases of every
quantized linear, the built backbone's dtypes and grad flags, and that the recorded
pass keeps no saved tensor (negatives: a weight-sized module, a fp32 base, plain
autograd); `tests/training/prewarm/test_prewarm_buckets.py` checks the replay points and the
attention calls stay within the planned maximum while covering every bucket
(negative: the bucket tops). `tests/kernels/tuning/test_tuner_working.py` checks the
working-memory bound (negative: no bound).

### Adapter autotuned-op prewarm

The adapter prewarm (`AdapterSet.prewarm`) settles each distinct `AutotunedOp` once
(docs/internals/adapter-binding.md#autotuned-ops). Ops whose `signature` matches (the
tuning name, i.e. op and backend, and the layouts of every operand without a token
axis) share their problems; the problem key is a function of exactly these, the token
bucket and the grad pattern. Within an op, a bucket and grad pattern already settled in
process or stored in this card's table (`lycoris_<op>_<backend>.json`) is bound without
timing, so a restart against the same kernel cache times nothing. The prewarm logs the
autotuned ops, the distinct ones, the problems measured and its device peak.

Memory follows the kernel prewarm's rules. The scratch inputs are constructed at the
token buckets capped at the planned maximum (`token_buckets`). Problems run one at a
time: each runtime input is built when its problem is reached
(`runtime.problems(rows)`: a builder per bucket and grad pattern) and dropped after
it, and a measured problem's free blocks go back to the driver (`release`) before the
next, so the cached blocks of one token count never stack under the next. Each
problem's body candidate is timed first (`select(lean=BODY)`), and every candidate's
timing, LyCORIS's own cold tile tuning inside the kernel's first call included, runs
under an absolute budget: `hardware.memory.tuning_budget`, the kernel prewarm's
device peak (the run's block ops at the planned maximum, below the training steps'
peak) minus the bytes allocated now, at most `BUDGET_FRACTION` (0.5) of the device's
free memory after the run's resident state. A cached kernel prewarm replays little
or nothing, so its ops' peaks sit near the resident baseline and would give a budget
near zero. The kernel prewarm therefore stores its working bytes (its device peak
minus the bytes allocated when it started) in the prewarm manifest under one key per
prewarm (`prewarm_working|<hash of its replay problem keys and attention key>`), and
`note_prewarm` keeps as its peak the peaks of the ops that ran together with the
baseline now plus the stored working bytes, storing the larger working back; a cached
op's near-baseline peak never counts. A fully cached run thus gets the budget of the
run that measured it. With nothing replayed and nothing stored (a manifest older than
the stored value) no kernel-prewarm peak is noted and the free-memory term applies
alone. Under several processes only rank 0 runs the kernel prewarm, while every rank
runs the adapter prewarm and times its own candidates (`training/entry.py`
`_prewarm`); rank 0 therefore releases the prewarm gate with its working bytes as
the gate's payload (`share_working` -> `release_ranks(gate, payload=...)`, the store
key the other ranks already wait on, no extra collective), and each other rank keeps
its bytes allocated now plus those working bytes as its prewarm peak
(`take_working`), so every rank's adapter budget is rank 0's. A token bucket over the budget is never timed and inherits instead (`tuner.inherit`): the decision of the largest measured
bucket of the same op and grad pattern, stored with `"inherited": <source key>` and
logged under `inherited`, so a live call at that size binds it and never tunes. A
bucket is over the budget when the largest working memory of its source bucket,
scaled by the token rows, passes it, or when a candidate's timing reaches the cap
(`select(cap=...)` raises `OverBound` before the allocation, nothing stored). The
smallest bucket of a pattern is always timed. The kernel stays the default where
nothing is timed, and the relative bound (8 x the leanest) applies as before. The
prewarm starts and ends on an empty allocator cache and its device peak enters
`note_peak`, so the step-1 line `VRAM peak: prewarm` and the `prewarm_peak_*`
metrics cover it.

The end of the adapter prewarm (`prewarm.go_capped`) makes that budget the
process's: `tuner.set_cap(device, budget)`, so every later `select` that passes no
cap (inside the tuning windows below: the between-step kernel-vs-compile retune, the
text-encoder prewarm in step 1, the first sampler step) times within it; a key whose timing would pass it
binds its first candidate for the process, untimed and unstored (`tuner.DEFERRED`,
one INFO line `over the tuning budget`). A live tune returns no memory to the driver:
its working memory stays in the allocator's cache for the steps to reuse (no `trim`
inside the training loop). It also turns LyCORIS's tile
tuning off (`set_lycoris_tuning(False)`, restored at the run's end): a LyCORIS key the
prewarm did not tune (an inherited bucket's size) binds LyCORIS's stored or
model-ranked tile and is never tuned inside a step. The adapter prewarm flushes its
decisions as their own table (`after the adapter prewarm`), so the first retune's
table lists only the retune's decisions.
`tests/kernels/tuning/test_tuner_working.py::test_live_tunes_stay_within_the_budget`
(negative: no cap) and
`tests/training/prewarm/test_adapter_prewarm.py::test_prewarm_end_caps_every_later_tune`
(negative: LyCORIS left tuning) check it.

### Tuning windows

A training run tunes kernels only in these windows:

| window | what is timed | memory returned to the driver |
| --- | --- | --- |
| the prewarm before step 1 (and `kohakumoss-tune`) | every planned problem | before and after each phase, before each bounded config |
| the online text encoder's prewarm, right before its first pass (`TextPrewarm`) | its pack buckets | after it |
| the first `RETUNE_STEPS` steps of the process (default 3), between steps (`RetuneKernels`) | queued kernel-vs-compile problems (`torch.compile` against the kernel) | only when a queued problem's inputs do not fit |
| the first sample grid of the run, up to the end of its first sampler step (`PipelineImages`) | missing configs of the sampling shapes | the sampler's own releases (after encode, after sampling, after decode) |

The window state lives in `kohakumoss_kernels.tuning.window`. The end of the prewarm
(`go_capped`) calls `window.freeze()`: outside a window,
`select` times nothing, and a key first met there binds the first candidate that builds
and runs once, untimed, unstored, named in `tuner.DEFERRED` and logged once at WARNING
(`met after the tuning window`): the prewarm's plan did not reach that shape, and the
next run's prewarm covers it once the plan does. `window.tuning_window()` lifts the freeze
inside the text-encoder prewarm and the first sampler step, and `timing_here()` lifts
it inside the retune. After the last retune step `RetuneKernels` calls `window.seal()`:
a kernel-or-compile op no longer queues a new problem (it binds the kernel), the
problems still queued are dropped and stay on the kernel (one INFO line `kernel tuning
closed after step N`), and the callback is a no-op from then on. The count is per
process, so a resumed run retunes its own first steps; the run's end reopens the
windows (`window.reopen()`), so a later inference or tuning call in the same process
tunes as a fresh one. Under a process group a window after step 1 lifts the freeze, not
the go-live deferral, so its missing keys stay untimed there. The VAE decode of the
first sample grid runs after its first sampler step and binds untimed where a key is
missing. No window returns memory to the
driver on its own schedule inside the loop, and no step outside them tunes.
`tests/kernels/tuning/test_tuning_window.py` checks that a frozen key binds untimed
and a window times it (negative: a select that ignores the freeze), that the retune
runs over the first N steps only and then seals (negative: a retune every step), that
a sealed op queues nothing (negative: queued after the seal), and that only the first
sample grid's first sampler step is open (negative: every grid open).

The kernel candidate is the heavy one where the input takes a gradient. On an RTX 5090
(LoKr full_matrix factor 16, fp16, the 3072 -> 16384 bypass with base at 20000 rows)
its timing works on 6680 MiB against 2500 MiB for the compiled body, and 5472 against
2048 MiB at 16384 rows, 3040 against 1024 MiB at 8192; the kernel is the faster one
from 4096 rows (5.6 against 9.1 ms at 20000). Held in the allocator cache from one
problem to the next, those blocks stacked to 12.8 GB in nvidia-smi (11.5 GB on
LyCORIS 4.1.2) against 6.6 GB for the training steps and a 4.0 GiB kernel prewarm
peak; with that peak as the budget, the 16384- and 20000-row buckets inherit the
8192-row decision (the kernel). `tests/training/prewarm/test_adapter_prewarm.py` checks that
only one problem's inputs are alive while it is timed (negative: every bucket's
inputs built up front), that an over-budget bucket inherits the largest fitting
bucket's decision without being timed, by prediction and by the cap (negative: no
budget, the bucket is timed), that a fully cached kernel prewarm gives the budget of
the uncached run and a never-measured key the free-memory term (negative: the stored
working bytes ignored), and that the reported peak holds the adapter prewarm
(negative: the peak not noted).

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
Triton specializes integer arguments only on divisibility by 16 and equality to 1.
A new value inside a bucket therefore reuses the compiled kernel.
The keys carry a version tag (`KEY_VERSION`), so rows stored under an older key
format are never matched.

`kohakumoss.encoders.text.prewarm.prewarm_text` records the kernel calls of one
synthetic two-row pack through a weightless twin of the encoder's text model
(`text_shapes`: the model class built on the meta device from the encoder's config,
the plan's quantized linears, each tensor in the dtype of the encoder's tensor of the
same name, materialized as stride-0 views, quantized GEMMs bound to the device), run
on the encoder's own ops under no_grad; the placed weights are never read. It replays
the calls at one target pack per bucket the plan can produce: packs of at most
`plan.batch` rows and `plan.max_tokens` tokens, rows up to
`min(MAX_TEXT_TOKENS + chat template, encoder max length)` and, when the trainer knows
the data, at most the length bucket of the longest caption estimate plus the template
(`longest_row`; `MossDataset.text_tokens_max`, the planner's estimate columns; the
bucket rounding covers an estimate a little below the tokenized length, and a longer
caption is tuned on its first encode). `pack_lengths` builds the row lengths. For every (segments, longest, mean) bucket it builds one long row plus
equal shorter rows whose mean lands in the bucket. It then adds packs for every
token bucket not yet reached. The skewed packs (one long row and many one-token rows)
are where the padded PyTorch delta-rule path is at its largest; the working-memory
bound keeps it from setting the tuning peak. `tests/inference/test_text_prewarm.py`
checks on CPU that the planner covers the buckets of random `plan_groups` packs
(negative: equal rows only) and that the data cap bounds the rows (negative: no cap).

## Multi-process

`tuning.broadcast(src=0)` replaces every rank's tables with rank `src`'s (memory and
disk) over the initialised process group, then runs `tuner.reset()`: settled choices and
every binding cache registered with `tuner.on_reset` (kernel-or-compile block ops,
fused-linear runners, attention, text-encoder, optimizer and quant bindings) are
dropped, so live calls bind from the shared
tables. The trainer prewarms on rank 0 only, broadcasts, then starts; with identical
cards every rank runs the same configs (a rank on a different card has no rows for its
card and tunes its own). The text-encoder prewarm runs on each rank before that rank's
first encode, without a broadcast (a rank whose captions are all cached never encodes);
each rank's text encoder encodes only its own prompts.

Problems timed after the prewarm (inside the tuning windows; the adapter and
text-encoder prewarms run on every rank) are timed by the rank that meets them.
After each retune step, on every rank, `RetuneKernels` runs `tuner.agree()`: each rank's
measurements since the last step are counted with one one-element all-reduce
(`measured_anywhere`, on the backend's device) and gathered (`all_gather_object`) only
when some rank measured something, so a step after warm-up costs one small
all-reduce; the lowest rank's measurement of each (card, kernel,
key) wins (`agreed`), and every rank with a device of that card settles it, stores it,
and drops its bindings when the winner replaces a different local choice (`adopt`), so
every rank binds the same config from the next step on. One process (or a
single-rank group) skips the exchange. `tests/kernels/tuning/test_tuner_agree.py` checks the
Inside a step under a process group of two or more ranks no key is timed: from
`decisions.go_live` (the end of step 1) on, `tuner.defer_if_grouped` makes `select`
bind the first candidate of a missing key that builds and runs once, untimed (a
config raising a build error is skipped as in timing; all failing raises
`ConfigsFailed`), for the process, unstored, named in `tuner.DEFERRED` and logged once
at INFO (`deferred under DDP: first candidate, untimed`) (its peers would otherwise wait in the step's collectives under the
NCCL timeout); the between-step retune times inside `window.timing_here()`. A sample
grid's first sampler step (rank 0 only) stays deferred under a process group: its
window lifts the freeze, not the deferral. The retune times only kernel-vs-compile
problems, whose call layouts it queues; a deferred generic `select` key (attention,
quantized GEMM, adapter configs) keeps no call to replay, so under DDP it stays on its
first candidate for the run, and the next run's prewarm covers it when its plan
reaches that shape. The test checks the deferral, the single-process
path and the retune (negative: no deferral, timed in the step), the
merge (negative: the highest rank wins), the rebinding (negative: bindings kept) and
that the objects are gathered only after a measurement (negative: a gather every
step).

`KOHAKUMOSS_KERNEL_CACHE` moves the cache root (default `~/.cache/kohakumoss_kernels`).
The test session points it at an empty temporary directory before any table is read
(`tests/conftest.py`), so every test runs one fixed config per op and never reads the
host's tables; a test that tunes sets `table` and its own cache directory.
