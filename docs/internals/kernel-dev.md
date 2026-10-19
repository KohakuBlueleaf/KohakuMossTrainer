# Kernel development

`kohakumoss_kernels` holds the MOSS-block ops. Each op has one eager reference
(`backends/eager/`, defining what the op computes) and per-arch implementations:

| arch | implementation | why |
| --- | --- | --- |
| sm70 / sm75 (V100, 2080 Ti) | TileLang | Triton has no tensor-core dots below sm80 (they lower to FMA) |
| sm80 / sm86 / sm89 / sm120 | Triton (TileLang also registered) | Inductor sees `triton_op` launches |
| XPU | Triton (Intel backend), per op where measured faster ([performance](../performance/README.md)) | TileLang has no XPU target |
| CPU | eager | |

Every tuned kernel family times a PyTorch path inside its shortlist, and the winner is
stored per card, so a kernel is bound only where it beats PyTorch:

| op family | PyTorch path |
| --- | --- |
| block ops, fused linears | `compile` (torch.compile of the eager reference), `backends/compile/choice.py` |
| `attention_varlen` (Triton and TileLang entries, `backends/varlen_choice.py`) | torch `varlen_attn` where the build and card have flash, else the per-segment SDPA loop |
| `attention_varlen_causal` (text encoder) | torch `varlen_attn` (causal) where available, else the per-segment SDPA loop |
| `gated_delta_rule_varlen` (text encoder) | the chunked eager reference (`torch` config) |
| `quant_linear` | dequantize + matmul (`quant.PYTORCH_PATHS`) |
| `fp8_linear*` | the vendor `torch._scaled_mm` path and the 16-bit dequantized paths (`W16`) |
| optimizer steps (`adamw_step`, `adamw8bit_step`, `adamw_step_multi`) | the reference step, timed once per key on scratch copies (`optim/choice.py`) |

On XPU the Triton fused linears (`linear_situ`, `linear_gated_residual`,
`linear_qk_rope`) and the `fp8_linear*` paths are registered too:

- Each tile that fails to build on the Intel backend is skipped by the tuner.
- A problem with no surviving candidate raises `BackendUnavailable`, and the registry
  rebinds the op to its next backend (`compile` for the fused linears, `eager` for fp8).
- Without fp8 matrix units the fp8 paths that dequantize in registers (`promote`,
  `fused`) and the 16-bit paths remain.

## Registry

An implementation is a callable with the op's eager signature (`kohakumoss_kernels.ops`)
that supports autograd and torch.compile.

| call | does |
| --- | --- |
| `register(op, backend, fn, supports)` | adds an implementation; `supports(query)` returns None when it can serve the `OpQuery` (op, arch, dtype, static options), or the reason it cannot |
| `resolve(op, device, dtype, **static)` | walks the backend priority of the device's arch (`DEFAULT_PRIORITY`, overridable with `set_priority`), skipping backends that are unavailable, do not support the query, fail their probe on the device or have a recorded failure for the op, and returns a `Bound` op |
| `available(...)` | lists every backend's reason (the doctor's view) |

Inside `recording()` (`kohakumoss_kernels.recording`) a `Bound` call is snapshotted and
answered by the eager reference, so nothing is built or tuned; see
[tuning.md](tuning.md#prewarm).

## Adding an op to a backend

1. Write `backends/<backend>/<group>/<op>.py` (groups: `blocks/`, `attention/`,
   `textenc/`, `optim/`, `fp8/`). The backend package imports every public module of
   its package tree on import (files starting with `_` are helpers and are skipped), so
   no `__init__` edit is needed.
   - A submodule whose import raises `ImportError` (a missing optional module) is
     listed in `backends.<backend>.FAILED` and logged at WARNING; the rest still
     register.
   - Any other error raised while importing a submodule (a `NameError`, `TypeError`,
     ... in its code) propagates.
   - `tests/kernels/registry/test_backend_imports.py` asserts `FAILED` is empty for
     every backend whose package is installed.
2. At module bottom call `registry.register(op, backend, fn, supports)`, with
   `supports` built from `constraints.requires(...)`. `fn` has the eager signature,
   supports autograd and `torch.compile(fullgraph=True)`, and handles every shape it
   accepts (resolution happens once per build, never per call).
3. torch.library wiring:
   - Triton: `torch.library.triton_op` + `register_autograd`; launch through
     `backends.triton._common.Launcher` (direct on real tensors, `wrap_triton` while
     traced) so Inductor records the launches. Outside torch.compile the block ops call
     an `autograd.Function` over the same forward / backward bodies and setup
     (`_common.eager_function`): the custom-op dispatch costs a few hundred us of host
     time per forward + backward, which decides small problems (N of a few thousand
     tokens on sm86+), and the Function computes the same thing without it.
   - TileLang: `torch.library.custom_op` + `register_fake` + `register_autograd`.
     Allocate outputs with torch and pass them in (`_blockops` launches take
     caller-allocated outputs): tensors TileLang allocates lose autograd when a custom
     op returns them.
4. Tiles come from `tuning.select` (see [tuning.md](tuning.md)).
5. Tests in `tests/kernels/`: the error-statistics check of `tests/_util/error_stats.py`
   in fp16 and bf16, forward and backward, plus a broken variant the check must reject
   (`tests/_util/negative.py`).
6. Benchmark in `scripts/bench/kernel/` against every PyTorch alternative, eager and
   compiled, accuracy next to speed ([Benches](#benches)).

## Per-sample vectors

The segment-indexed ops (`rms_norm_mod` shift / scale, `gated_residual` and
`linear_gated_residual` gate) receive `mod[:, k]` slices of the backbone's fp32
`[B, 6, D]` block modulation and `[B, 2, D]` head modulation: row stride `6 D` (or
`2 D`), not `D`.

- The kernels take the row and column strides of each per-sample vector as arguments and
  address `seg * row_stride + col * col_stride`; no copy is made.
- With a single sample every row reads row 0, so only packed batches of two or more
  samples expose a layout error.
- `tests/kernels/blocks/test_modulation_layout.py` runs every backend on the strided
  slices against its own result on contiguous copies.

## TileLang notes (0.1.15)

- Token and segment counts are `T.dynamic`; one compile per (heads, head dim, dtype,
  tile config, arch) serves every batch. A grid extent can be a scalar argument.
- `T.copy` from a symbolic-shaped tensor bounds-checks rows past the end (zero fill);
  element stores to symbolic tensors are bounds-checked too.
- Use slice regions (`X[a : a + bm, head, :]`) in `T.copy`, not start indices, on
  tensors of rank > 2.
- Moving a GEMM accumulator into the A operand of the next GEMM goes through shared
  memory. A fragment-to-fragment copy between the two layouts fails layout inference
  ("Layout infer conflict") on every arch, and Volta's MMA has no transposed-A form.
- Warp partition: each warp owns a multiple of 16 rows and 16 columns of every
  `T.gemm` tile (checked before compiling).
- Loops whose trip count depends on the program use `T.serial` (no pipelining).
- On Windows nvcc accepts only MSVC; `hardware.toolchain.ensure_toolchain` (run on the
  first kernel build, see [Lazy backends](#lazy-backends)) imports the `vcvars64.bat`
  environment and sets `CXX` to that `cl.EXE` (see
  [install-windows.md](../guides/install-windows.md#compiler-setup)).

## Fallback

`registry.resolve` returns a `Bound` op that calls its implementation directly. An
implementation that cannot run here raises `errors.BackendUnavailable`:

- the tuner raises `ConfigsFailed` (a subclass) when every shortlisted config fails to
  compile or run;
- a kernel-or-compile op (`backends/compile/choice.py`) raises it when its
  once-per-device probe fails;
- the compile backend raises it when an op's first compile on a device fails;
- the attention and text-encoder entries translate a build / launch error of their call
  into it (`errors.unavailable`).

Two errors are not a backend failure:

- When the eager reference also rejects the failing call's arguments, `Bound` raises
  `errors.InputError` instead: a caller bug, nothing recorded, the op stays bound.
- A sticky device error (`errors.is_sticky_error`: the text names an illegal memory
  access, misaligned address, illegal instruction, device-side assert or unspecified
  launch failure, or synchronising the CUDA device raises) propagates unchanged, with
  nothing recorded and no rebind: the context is unusable, so any next backend would
  fail on it too (`tests/kernels/registry/test_fallback.py`, negative: the check off).
  The prewarm replay re-raises it as well instead of skipping the problem
  (`tests/kernels/tuning/test_replay_sticky.py`).

### Rebind

Otherwise `Bound` records the failure in `registry.failures()` and rebinds to the next
backend of the arch priority, together with every live `Bound` of the same query (op,
arch, dtype, static options, priority) still on the failed backend: the prewarm twin's
op and the model's op are such a pair, so a failure met in the prewarm moves the
model's op too. Each rebind logs one WARNING through the project logger:

```
<op> on <arch>/<dtype>: backend <failed> failed, using <next> (<n> bound ops): <error>
```

A pinned backend (`resolve(..., backend=...)`) re-raises instead.

### Failure records

- **Key.** A record is keyed `op|backend|arch|dtype|static options`
  (`status.failure_key`), plus the problem (`<tuned kernel>:<key>`, the key holding the
  bucketed shapes) when the error names one: a `ConfigsFailed` (every shortlisted config
  of one problem failed) is a record of that problem only.
- **On disk.** Only a build / toolchain failure (`is_toolchain_error`) met at build time
  is written to `<kernel cache>/failures/backends.json`: inside `status.build_phase()`,
  which `registry.exercise` (the prewarm replay, `tuning/replay.py`, and the attention
  prewarm, `tuning/prewarm.py`) enters. A failure met in a live call (a step, a sample) rebinds the op in this
  process and is never written to disk.
- **Scope.** `resolve` skips a backend for an op only on a record without a problem for
  the same dtype and static options, so a failure of one shape leaves the backend for
  every other shape, and a failure in fp16 leaves bf16.
- **Visibility.** A backend skipped for a record on disk logs one WARNING (`backend
  <name> skipped, failed here (...)`) when the op resolves, so a run start shows every
  persisted demotion.
- **In process.** The tuner raises the stored `ConfigsFailed` of a problem again on its
  next lookup without timing it again.
- **Environment.** A recorded failure is stored with the environment fingerprint
  (package versions, toolchain, Intel source mode) and honoured by later processes with
  the same fingerprint; a changed environment ignores it. `kohakumoss-doctor` lists the
  probe results and the current-fingerprint failures; `kohakumoss-doctor
  --clear-failures` forgets them.

### Backward failures

The fallback covers failures raised while the forward runs, so a backend surfaces its
backward failures there:

- the attention kernels build and launch their backward once (zero `dO`, no grad) on the
  first grad-enabled forward of a problem key, inside the same `BackendUnavailable`
  translation as the forward, so a backward that cannot build or launch hands the op on
  before any activation was saved on that backend; under `TUNE=off` the untimed pick
  (the stored winner, else the first) is the one launched;
- `registry.exercise(op, run)` runs a forward + backward (the prewarm replay does this
  for every recorded call at every planned bucket, the attention prewarm for every
  length and segment-count bucket) and falls the op back on a failure raised anywhere
  in it, the backward included.

A backward failure that still reaches the backward pass (a new key first met inside a
checkpoint recompute, where `_Guarded` does not run) raises there.

**Guarded calls.** Until one backward of the current backend has succeeded, a `Bound`
call that records a plain autograd graph (grad on, not traced, no saved-tensor hooks, a
tensor needing grad) runs through `registry._Guarded`:

- the implementation runs on detached leaves and its backward through `autograd.grad`,
  so a failure raised only in the backward also falls back and the gradient is
  recomputed on the next backend;
- the recompute reads the call's tensor inputs from `save_for_backward`, so an input
  modified in place since the forward raises autograd's version error instead of
  recomputing on the new values;
- checkpointed and offloaded calls (saved-tensor hooks) keep their own graph semantics
  and are not guarded;
- the backward releases the forward's leaves, outputs and arguments (they are kept
  as `ctx` attributes, which autograd does not free): a graph that outlives its
  backward (a driver holding the step's loss into the next step) keeps none of them.
  Without the release, an uncheckpointed step 1 keeps its whole activation set into
  step 2's forward;
- a good backward marks the op proven, and `registry.settle_guards()` at the step
  boundary (`finish_step`, `MemorySession.step`) binds its plain path, so one step
  (every micro-batch of it) runs one call path per op. Outside a step
  (`registry.exercise`, the prewarm) a good backward unguards the op at once.

The drivers also return each step's `StepOut` detached (`StepOut.detached()`), so no
step's graph is held past its backward whatever else keeps `ctx` attributes.

**Call paths.** A `Bound` binds its call path once per state instead of testing it per
call:

| state | path |
| --- | --- |
| inside `recording()` | the recording path |
| `guard` on | the guarded path (the per-call `_records_grad` test) |
| otherwise | the implementation directly inside the fallback's `try` |

Setting `guard` (`settle_guards` after a good backward, a fallback) and entering or
leaving a recording (`recording.on_change`) rebind every live `Bound`.

### What counts as a failure

"Cannot run here" is `errors.BUILD_ERRORS` (RuntimeError, OSError, UnicodeError,
Triton's `TritonError`), never out-of-memory: an OOM or a bug propagates and disables
nothing.

**Backend probes.** Before any op binds a backend on a device, `resolve` runs that
backend's probe once per process and device (`status.register_check`):

| backend | probe |
| --- | --- |
| Triton | builds and launches one minimal kernel (this also initialises Intel's driver utilities on XPU) |
| compile | inherits Triton's result on a GPU; needs the host C++ compiler on CPU |
| TileLang | needs the host compiler for nvcc |

A failed probe logs one WARNING and the backend is skipped for every op on that device,
so a missing compiler costs one build, not one per op.

**Text-encoder ops.** The text-encoder ops (`encoders/text/forward.py`, `TextOps`)
follow the same rule at their build-time probe:

- a RuntimeError (out-of-memory included) binds the eager reference with one WARNING
  (`text encoder op <name> on <device>: backend <b> failed its probe, using eager:
  <error>`), kept in `TextOps.fallbacks`, never written to disk; any other error
  propagates;
- an out-of-memory in the probe is retried once after the device's free blocks go back
  to the driver (`forward.probe`), since the eager reference usually needs more working
  memory than the kernel and a transient OOM would otherwise bind it for the run
  (`tests/kernels/textenc/test_textenc_probe.py`, negative: no retry).

## Lazy backends

`import kohakumoss_kernels` does no backend work, so a data helper or a CLI that never
builds a kernel pays none of it. The first kernel build (`registry.resolve` or
`available`, i.e. the first `OpSet.resolve` of a model) runs `registry.load_backends()`
once per process:

1. it applies the process's compile settings (`compile_setup.apply()`: the
   compile-cache key tag, [Compile cache key](#compile-cache-key), and the compile
   worker pool, [Compile backend](#compile-backend));
2. it imports the eager, compile, Triton and TileLang backend packages; the Triton and
   TileLang packages set up the host toolchain on import (`ensure_toolchain()`:
   `vcvars64` and `PATH` / `CXX` on Windows, the Intel Triton UTF-8 sources), the
   compile backend at its first CPU probe;
3. it runs one cyclic gc pass. The compile backend imports `torch._dynamo`, and that
   import leaves a reference cycle through the importing stack (`torch.fx.wrap` keeps
   its own frame, whose `f_back` chain reaches every caller frame): without the pass,
   the frame of the call that built the first kernels, `load_backbone` with the model
   among its locals, outlives it until the collector runs. See
   [reference-cycles.md](reference-cycles.md).

`CompiledOp` also applies the settings before its first compile.
`tests/kernels/registry/test_lazy_backends.py` checks it in a fresh process (negatives:
backends loaded at import; the gc pass left out, which keeps an object held by the
first build's caller alive with automatic gc off).

## Current device

Kernels launch on the current device's queue, so every Triton launch (`Launcher`) and
TileLang launch (`_blockops.compiled`) runs under `hardware.current.on_device(...)` for
its tensors' device; without it a tensor on `cuda:1` / `xpu:1` launched while device 0
is current reads the wrong device (DEVICE_LOST on Intel Arc).

`hardware.current` also holds:

| name | role |
| --- | --- |
| `synchronize(device)` | the one device-wait helper of the package |
| `ACCEL`, `is_accel`, `device_module` | the accelerator device types |
| `saved_tensor_hooks_active()` | the one probe of torch's saved-tensor hook stack (a checkpointed forward or its recompute, activation offload) |
| `rng_kept(devices)` | `torch.random.fork_rng` over the CPU and the given accelerator devices |

`rng_kept` wraps every random draw tuning makes: `select`'s shortlist timing, the
prewarm replay's materialized inputs, the attention and adapter prewarms,
`retune_pending`'s rebuilt inputs, the device probe, the recorded twin pass and the
text-encoder op probe. The caller's generators are where they were after it, so a run's
random stream (adapter init, noise, timesteps) never depends on what the kernel cache
held; `tests/kernels/tuning/test_rng_kept.py` checks it (negative: the draws without the
fork).

## Compile backend

`backends/compile` registers torch.compile of every block op's eager reference and of
the eager compositions of the fused linear ops (`compile/ops.py`, `COMPILED`). Kernel
backends choose against it per shape (`compile/choice.py`); its probe on a GPU is
Triton's (inductor's GPU code is Triton), on CPU it needs the host C++ compiler.

**Call patterns.** A `CompiledOp` keeps one compiled callable, with its own code object
and Dynamo cache, per call pattern: what Dynamo guards on besides shapes.

- The pattern is the device, grad mode, intra-op thread count, autocast on the
  arguments' device, and per argument tensor dtype / requires_grad / rank, None and
  Python scalars.
- Dynamo recompiles per device, so a threaded run over several GPUs gets one callable
  per device instead of piling their graphs onto one pattern.
- The thread count is part of it because a checkpoint recompute runs on the autograd
  thread, whose intra-op thread count differs; the recompute then has its own compiled
  callable instead of recompiling the forward's.

**Limits and failures.**

- A pattern that needs more than `GRAPH_LIMIT` graphs raises `GraphLimitError`;
  Dynamo's own recompile limits are raised above it, so Dynamo's silent eager fallback
  is never reached.
- A build failure of the first call raises `BackendUnavailable`.
- Inside an outer torch.compile the eager reference is traced.

**Binding.** `CompiledOp.resolved(args)` is the runner of `args`' pattern. A
kernel-vs-compile op binds it per call key (the device, grad mode, thread count and
autocast dtype, then every argument's shape / dtype / grad flag or value, which holds
the pattern), so a bound call computes one key and calls the runner, not the pattern as
well. The adapter compile candidate (`adapters/bind/autotuned.py`, `CompiledBody`) binds
the `CompiledOp` of an autotuned op's operand shapes once per bucket
(`CompiledBody.at`).

**Compile worker pool.** On Linux inductor compiles Triton kernels in a sidecar
process with a pool of forked workers (`min(32, cpu count)`); on Windows it compiles
in-process. Inductor's default quiesces that pool 60 s after its last compile result
and forks a new one on the next compile, from a sidecar whose previous pool's threads
are still winding down; a worker forked then can deadlock on an inherited lock, and the
job it took never returns (`future.result()` has no timeout: the process waits
forever). `compile_setup.apply()` sets `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0`
(an environment value already set wins), and puts the value inductor's config reads
from it in place when that config is already loaded (a plain config assignment holds
for one thread only), so the pool is created without its idle Timer and lives for the
process. Every torch.compile site in `src/` calls it first
(`tests/rules/test_compile_sites.py`), the run line and sample.py log it
(`compile pool: no quiesce`), and `tests/kernels/registry/test_compile_pool.py` checks
the pool in a fresh process (negative: inductor's default pool has the Timer). Cost:
the workers stay up for the process; during an integration training run the 32
workers held 0.85 GiB proportional (PSS) host memory, 12.6 GB RSS counting the pages they share with the
sidecar, and the sidecar 0.46 GiB, which it holds either way. No device memory.

## Fused linear ops

`linear_situ`, `linear_gated_residual` and `linear_qk_rope` (GEMM + MOSS epilogue) run
one host code for both kernel backends, `backends/_fused_linear.py`: `FusedLinear`
defines a backend's six custom ops (`kohakumoss::<backend>_linear_{situ,gres,qk}_{fwd,
bwd}`), their autograd (save only what the backward needs while autograd records; the
backward is the epilogue's backward kernel, then cuBLAS GEMMs for the input and weight
gradients), the per-problem choice between `fused` (the backend's GEMM + epilogue
kernel, one entry per tile config) and `split` (cuBLAS + the standalone epilogue
kernel), `force_path`, and the public functions the registry binds. A backend module
holds only its fused launchers and tile configs and a `FusedBackend` record of what
differs:

| field | Triton | TileLang |
| --- | --- | --- |
| `define` | `triton_op` (traced by Inductor) | `custom_op` + the shared meta functions |
| `pick` | `tuning.select` over every config | `_blockops.tuned` (configs that failed to compile skipped) |
| `fusable` | every problem | no bias; `linear_gated_residual` also needs the residual stream in the activation dtype (else `split` only) |
| `dead` / `mark_dead` | none | a fused config raising `CompileError` is dropped and the problem picks again |
| `fallback` | none | `CompileError` from the public function runs the compiled eager composition |

The problem keys are `<token bucket>-<K>-<N>-<dtype>-<flags>` for both backends
(`_fused_linear._key`; flags: bias / gate / residual scale / residual dtype / save as
the op has them). The qk entry passes the q / k norm weights as fp32 (one cast at the
op boundary; the weight gradient comes back in the parameter's dtype), and the qk
backward casts the norm-weight gradients to the saved weights' dtype, on both backends.
`tests/kernels/blocks/test_fused_linear_host.py` checks that both backends run the same public
functions and op bodies (negative: a backend keeping its own copy) and the fp32 norm
weights (negative: the storage dtype passed through); `test_fused_linear.py` and
`test_saved.py` check the numerics and the saved tensors on a GPU.

## Selective checkpoint

`CHECKPOINT="selective"` keeps the outputs of the matmul and attention ops and
recomputes the rest (`kohakumoss/memory/runners.py`). The kernel backends launch their
attention and fused GEMM + epilogue forwards as custom ops, which the checkpoint's
dispatch mode sees under their own names, not as aten matmuls; each backend adds those
forwards to `kohakumoss_kernels.saved_ops.KEPT` at import (`attn_varlen_triton`,
`attn_varlen_tilelang`, the `*_linear_situ_fwd` / `*_linear_gres_fwd` /
`*_linear_qk_fwd` ops of Triton and TileLang, and torch's `torch_attn::_varlen_attn` /
`aten::_flash_attention_forward` with the varlen entries), and the checkpoint policy
keeps every op in `KEPT` as it keeps aten `mm` / `linear` / SDPA.
`tests/kernels/autograd/test_saved_ops.py` checks the policy on those ops (negative: an empty
`KEPT`).

## Compile cache key

torch's on-disk compile caches (the FX graph cache and the AOT-autograd cache under
`TORCHINDUCTOR_CACHE_DIR`) key an entry on the traced forward graph. A custom op's
`register_autograd` backward and its schema are not part of that key, so a cached entry
built from different op code (for example a backward with one argument fewer) would
replay that code's backward and call the op with its argument list ("expected at most
10 argument(s) but received 11").

`kohakumoss_kernels.cache_tag.apply()` appends `kohakumoss-<sha1[:12]>` to
`torch.compiler.config.cache_key_tag` on the first kernel build (`load_backends`) and
before a `CompiledOp`'s first compile:

- The hash covers the `kohakumoss` package version plus the text of every module under
  `kohakumoss_kernels` that registers a custom op or its autograd (sources containing
  `custom_op(`, `register_autograd(` or `torch.library`).
- Any edit to such a module, or a version change, gives every compile cache entry a new
  key, and entries under other keys are never read.
- A tag the user already set is kept, and ours is appended after a `|`.
- Computing the tag reads those sources once per process, a few milliseconds.

## Peak windows

The allocator has one peak counter per device, and resetting it would drop the peak of
every other measurement in progress. `hardware.memory` measures in windows instead:
`open_window` / `close_window` (or `with window(device) as w`) give the peak allocated /
reserved bytes while the window was open. Every open, close and read folds the counter
into every open window of the device, then resets it, so a nested or overlapping window
never loses an enclosing window's peak. `memory.fold` is the only caller of
`reset_peak_memory_stats` in the project. `memory.MODULES` overrides the device module
per device type (tests use a fake allocator).

## Optimizer

`kohakumoss_kernels.optim` binds `adamw_step`, `adamw8bit_step` and `adamw_step_multi`
once (`resolve`); `optim.reference` defines what each computes. The trainer's
`adamw8bit_kernel` optimizer (`kohakumoss.training.optim.optimizers.adamw8bit`) steps
through `adamw8bit_step`; `adamw_step` and `adamw_step_multi` serve the kernel benches
and tests.

- **State.** AdamW keeps `m` and `r = sqrt(v)`, so the squared magnitude never reaches
  16-bit storage. The 8-bit variant stores m and r per block of `BLOCK` values as int8 /
  uint8 codes of a square-root companding of value / block absmax.
- **Writeback modes.** `nearest`, `stochastic` (bf16: add a random 16-bit integer to the
  fp32 bit pattern, then truncate) and `kahan` (a compensation buffer in the parameter
  dtype carries what rounding dropped).
- **Seeds.** The multi-tensor step seeds tensor t with `seed + t * SEED_STRIDE`.
- **Selection.** Each backend kernel is chosen against the reference per problem key
  (`optim/choice.py`); scratch copies of the arguments are made only when a key is
  actually timed.
- **Gradient dtype.** The gradient may come in its own dtype (a 16-bit parameter with an
  fp32 gradient) in both backends.

## Test budgets

A kernel test that builds its kernels on first use runs on an empty kernel cache per
session, so its first compile is part of its time on every run; such a test carries
`@pytest.mark.budget(N)` (the slowest measured lane plus about 50%) over the default,
and keeps its full candidate coverage:

| test | budget s |
| --- | --- |
| `attention/test_attention.py::test_error_stats`, `::test_empty_and_single_segments` | 60 |
| `attention/test_attention_infra.py::test_backward_failure_falls_back_in_forward` | 90 |
| `attention/test_attention_infra.py::test_forward_failure_falls_back`, `::test_tilelang_backward_reads_no_row_stats_past_the_last_token` | 60 |
| `attention/test_attention_choice.py::test_chosen_implementation_has_no_error_regression` | 30 |
| `textenc/test_textenc_ops.py::test_delta_candidates_precision` | 60 |
| `autograd/test_compile_blockops.py::test_fullgraph` | 30 |
| `blocks/test_norm.py::test_rms_norm_mod` | 30 |
| `quant/test_fp8.py::test_path_forward_and_dgrad` | 25 |
| `tuning/test_kernel_or_compile.py::test_prewarm_binds_every_op` | 20 |
| `tests/adapters/test_kernel_ops.py::test_every_candidate_matches_the_eager_body` (BOFT at 48 s on sm75) | 75 |
| `tests/training/prewarm/test_adapter_prewarm.py::test_prewarm_leaves_factors_untouched` | 35 |
| `tests/training/prewarm/test_adapter_prewarm.py::test_prewarm_keeps_one_problem_alive` | 20 |

Paths without `tests/` are under `tests/kernels/`.

## Benches

`kohakumoss_kernels.bench.blockops` is the block-op harness of `scripts/bench/kernel/`.
Per case:

- **Timed:** the backend kernel, the eager composition, torch.compile of it (static
  shapes) and the registered compile path (dynamic shapes), forward and forward +
  backward (best of N with the L2 flushed and clocks warmed, plus CUDA-graph replay).
- **Compared:** every output and input gradient through the error statistics against an
  fp64 run of the eager reference, with the per-statistic worse-list against the eager
  composition.
- An entry that chooses per shape reports its choice (`selected`).
- Rows go to `runs/kernel/<op>-<card>.json`.

## Varlen attention

`attention_varlen(q, k, v, cu_seqlens, max_seqlen, scale)` on packed `[N, H, D]`
tensors (any token and head strides, contiguous head dim for Triton).

- The grid's first axis covers the longest segment; programs past their segment's end
  exit, so no host work is needed per batch.
- The Triton kernels compute token offsets in int64 (a segment start times a row stride
  passes 2^31 elements past about 233k tokens of a qkv-strided view).

Numerics (three rules, applied in both backends):

1. **Two-plane row statistics.** The forward stores the raw score max `m` and
   `log2(l)` as separate fp32 planes `[H, N]`, not one log-sum-exp. A single
   `c*m + log2(l)` loses `log2(l)` once `c*m` is large.
2. **Shift before scale.** Probabilities are `exp2((s - m) * c - log2 l)`; `s - m` is
   exact near the row max, so the dominant probability is exactly 1 and
   `delta = rowsum(dO * O)` cancels.
3. **Scale on the accumulator.** dS is rounded to 16 bits unscaled; the softmax scale
   multiplies the fp32 dQ / dK accumulators at the end.

Design space (the tuner picks per arch and problem key; each variant is a separate
compiled kernel):

| kernel | knobs |
| --- | --- |
| Triton fwd | BM, BN, warps, stages |
| Triton bwd | strategy `fused` (dK/dV per key block, dQ through fp32 atomics into an `[N, H, D]` accumulator that the delta kernel zeroes as it computes `delta`, then one scale-and-cast pass) or `split` (dK/dV kernel + dQ kernel, no atomics); each kernel tuned separately, each strategy timed with its delta launch |
| TileLang fwd | BM, BN, threads, `qreg` (query tile in registers), `share` (K and V through one shared buffer) |
| TileLang dK/dV | BM, BN, threads, `kvreg` (K / V tile in registers), `pds` (P and dS through one shared buffer) |
| TileLang dQ | BM, BN, threads, `qreg` (Q / dO in registers), `share` (V then K through one buffer) |

The register / shared-buffer variants trade shared memory for registers so more
programs fit on cards with small shared memory per SM (Turing: 64 KiB).

The fp16 / bf16 MMAs use fp32 accumulation everywhere.

### Measured design per arch

Every kernel's candidates start with the arch's own measured seeds
(`ARCH_SEEDS[sm]` in `backends/triton/attention/varlen.py`, `SEEDS[(kind, sm)]` in
`backends/tilelang/attention/varlen.py`), then the shared seeds, then the planner's ranking;
the tuner times the first `SHORTLIST`. Under `TUNE=off` a stored winner binds, else the
first legal seed runs untimed, as it does in a trace.

| arch | source | fwd (BM, BN, warps, stages) | fused bwd | dK/dV | dQ |
| --- | --- | --- | --- | --- | --- |
| sm120 (RTX 5090, 99 KiB shared per block) | full sweep of 512 tiles per kernel at 2 MP x1, 1 MP x2, 512 px x8 | 128, 64, 8, 3 (235 TFLOP/s) | 16, 128, 8, 4 (200) | 32, 32, 4, 3 (209) | 128, 32, 8, 3 (241) |
| sm80 (A100, 163 KiB) | FA2's / Triton's A100 tiles (not measured here) | 128, 64, 4, 3 | 64, 128, 8, 2 | 64, 128, 8, 2 | 128, 64, 8, 3 |
| sm86 / sm89 (RTX 3090 / 4090, 99 KiB) | the same sweep on the 3090 (sm89 shares the table) | 128, 32, 4, 2 (76) | 16, 128, 8, 2 (63) | 16, 128, 8, 2 (65) | 128, 64, 8, 2 (78) |
| XPU | the shared seeds, tuned per key | | | | |
| sm70 / sm75 (TileLang) | measured per-arch seeds | see `SEEDS` | | | |

On sm120:

- The fused backward (dQ through fp32 atomics) beats the split one at every measured
  shape (11.6 vs 14.7 ms at 2 MP x1) and reaches 86 % of FA2's backward.
- FA2 and our forward both run at the card's measured MMA rate (233 TFLOP/s probe), so a
  single long segment ties with SDPA
  ([performance](../performance/attention.md#the-single-segment-ceiling)).
- The fused kernel holds BN = 128 keys with dK and dV in fp32 registers; larger query
  tiles than 16 rows lose to register pressure.
- TileLang builds on sm120 too; there it is reached only through the registry fallback.

sm80 seeds are a design point for A100-class cards; the tuner replaces them with
measured choices when a problem is first tuned.

**Registry entry per backend** (`backends/triton/attention/entry.py` on sm80+ / XPU,
`backends/tilelang/attention/entry.py`): a `VarlenChoice` (`backends/varlen_choice.py`)
that times, per problem key, the backend's own kernels against torch's `varlen_attn`
(when the torch build and card have flash) or else the per-segment SDPA loop, and binds
the winner (table `attn_varlen_impl_<backend>`).

- The key carries whether a backward runs through the call: `-g1` (grad mode on, an
  input requiring grad) is timed forward + backward on leaf copies, `-g0` (inference,
  sampling) the forward alone under no_grad, so a training run and an inference run on
  one card each bind their own winner.
- It times only outside inference mode, checkpointed forwards, recomputes and traces;
  inside torch.compile the SDPA loop is never used.
- Trainers call `tuning.prewarm_attention` in the prewarm before the first step, so every
  length bucket and segment-count bucket is decided ahead of the checkpointed training
  forward ([tuning.md](tuning.md#prewarm)).

The tile and strategy keys of both backends are
`<dtype>-h<heads>-d<head dim>-L<longest segment bucket>-S<segment count bucket>`
(`tuning.buckets.segment_bucket`: 1, 2, 4, then 8 and above as one): the grid is
(blocks of the longest segment) x heads x segments, so a pack of many short segments
and one long one fills the card differently from a two-segment pack of the same
longest segment.

**Call path.** The first call of a problem key reads the table, or tunes where timing
is allowed ([tuning windows](tuning.md#tuning-windows)), and binds the chosen kernels
into one launch callable; later calls look the binding up by key and launch.
Provisional picks made while tracing are not bound.

**Guards.**

- Masked key columns are `-inf` before the max (fully valid rows always have a finite
  max).
- Masked tiles zero `p` explicitly so out-of-range rows never produce `inf * 0`.
- Empty segments launch nothing.
- The backward kernels read row statistics (`m`, `log2 l`, delta) only for rows inside
  the segment; rows past the segment end read 0 (explicit `if_then_else`, independent of
  TileLang's automatic bounds guards).

### Matmul precision and error statistics

Every MMA takes 16-bit operands with fp32 accumulation, as flash attention does: P is
rounded once before P·V and P^T·dO, dS once before dS·K and dS^T·Q, delta comes from the
16-bit O, and the accumulators, row statistics and softmax stay fp32 until the single
rounding at the store.

The tests check o, dQ, dK, dV through the error statistics
(`tests/_util/error_stats.py`):

- Kernel and PyTorch are both measured against fp64 (cosine; abs and rel error as mean /
  median / min / max).
- Every check appends its table to `$KOHAKUMOSS_PRECISION_DIR/attention.jsonl`.
- A test fails only on a regression: non-finite, or `MAX_ERROR_RATIO` x PyTorch or more
  on the cosine deficit or the mean / median errors.
- "PyTorch" is plain PyTorch as it computes the op: `pytorch_sdpa` (`tests/kernels/attention/_attention_case.py`), one
  `torch.nn.functional.scaled_dot_product_attention` call per segment in the inputs'
  16-bit dtype with gradients by autograd, on whichever SDPA backend the device
  selects.

The measured stats per arch are in
[performance/attention.md](../performance/attention.md#precision).

**Huge logits** (clustered q / k, logits ~1e5). The softmax is nearly one-hot and
`sum_j dS_ij = 0` holds only before rounding. dQ = `scale * sum_j dS_ij K_j`, and K
carries a large component common to all keys, so rounding dS to 16 bits (bf16: 8 bits)
leaves a residue `(sum_j round(dS_ij)) * K` that the true dQ cancels. Every
16-bit-operand implementation shares it (flash SDPA included); the huge-logit stats are
in [performance/attention.md](../performance/attention.md#precision).

The test pools 16 draws (a handful of elements carry each draw's error) and applies the
regression check to the cosine and absolute stats only: with a one-hot softmax most fp64
dQ / dK elements are around 1e-190, far below any 16-bit dtype's range, so their
relative error (mean and median up to 1e18 - 1e91 for flash SDPA and ours alike)
measures the reference, not the kernel. The relative stats are still written to the
table.

`torch.compile` (fullgraph) of the kernels is compared with the eager call through the
error statistics, not bitwise: Inductor rebuilds the launched Triton kernels with its
own compile options, which may contract operations differently.

Bitwise repeatability per backend and strategy (repeat calls on the same inputs and
card):

| path | o | dK, dV | dQ |
| --- | --- | --- | --- |
| Triton fwd + `split` bwd | bitwise | bitwise | bitwise |
| Triton `fused` bwd | bitwise | bitwise | not bitwise (fp32 atomic adds in arrival order) |
| TileLang fwd + bwd (delta, dK/dV, dQ; no atomics) | bitwise | bitwise | bitwise |
| torch `varlen_attn` (timed candidate) | as torch | as torch | as torch |

The Triton strategy is a tuned choice, so dQ is bitwise only when `split` wins. A
different tuned tile config gives different (equally accurate) bits.

**Eager bounds.** The eager reference runs one SDPA per segment and needs the segment
bounds on the host. It takes them from a cache keyed by the `cu_seqlens` tensor:

- `SeqInfo.to` registers the bounds it already holds (`remember_bounds`) when it moves
  the plan to the device, so the eager path never syncs; an unregistered tensor is read
  back once.
- An entry is used only while the tensor lives and its version counter is unchanged, so
  an in-place rewrite or a new tensor at a reused address reads back again.
- Inference tensors carry no version counter and are read back on every call.
