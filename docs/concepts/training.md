# Training

`scripts/train/moss.py` is the training script: every knob is an UPPER_CASE global,
set in a config or on the command line:

```bash
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set LR=2e-4
```

`kohakumoss.training.entry.run(globals())` validates every knob at once (one
`SystemExit` listing every problem, with "did you mean" hints for typos), then builds
in a fixed order and hands the loop to a driver. The walk-through is
[first-lora.md](../guides/first-lora.md); this page is what the trainer does.

## Build order

1. Precision (`PRECISION`, `KEEP_FP32`, `FP16_FIX`, `FP16_FIX_KWARGS`).
2. Backbone: the `WEIGHTS` checkpoint, loaded straight to the device.
3. `FP16_FIX` (`apply_fp16_fix`), then the memory plan (quantized base, weight
   streaming, activation offload, block runner).
4. Trainable set: LyCORIS adapters (`kohakumoss.adapters`), or `FullTrainable` for
   `ALGO="full"` (the backbone's own parameters; the text refiner stays frozen unless
   `TRAIN_TEXT_REFINER`). Trainable parameters move to the master dtype.
5. `bind_ops()`, encoders, kernel prewarm, pipeline stages, parallel hooks, compile,
   TrainStep, OptimBundle, EMA, data, logger, callbacks, checkpoints, resume, driver.

Everything is resolved here, once; the loop calls what the build resolved.

- **Fresh state per run.** Each run starts from the kernel library's process state of a
  fresh process (`kohakumoss_kernels.run_state.reset_run_state`: no in-memory backend
  failure or probe result, default backend priorities, no settled tuned choice, an
  empty decision log; the tuning tables and the on-disk failure record stay), so a
  second run in one process (a notebook, a sweep) does not inherit the first one's.
- **Encoder placement.** After the build the log names where the text encoder and the
  VAE were placed, in the inference placement lines, or why the run holds none.
- **Seeding.** The global RNG is seeded with `SEED` at the run's start and again right
  before the build, so the build's init does not depend on the dataset scan (identical
  model and adapter init on every rank), and with `SEED + rank` after it. The σ and
  noise generators of the train step are seeded from `(SEED, data_rank)`
  (`layout.data_rank(rank)`): each data-parallel
  rank draws its own σ and noise; under `PARALLEL="pipeline"` every stage uses rank 0's
  streams (all stages see the same micro-batches).

### Kernel prewarm

Before step 0 the backbone's resolved `attention_varlen` and block ops (norms, RoPE,
SiTU, gated residual) run forward + backward on scratch inputs over every length bucket
the planned batches reach. They run outside any checkpoint, `no_grad` or inference
mode, so no kernel choice is made inside a checkpointed forward.

| prewarm input | from |
| --- | --- |
| longest sample (attention segment lengths up to it) | `min(MAX_TOKENS, largest planned latent grid + max_text_tokens)` |
| largest batch | `MAX_TOKENS` for token batching, `BATCH_SIZE` samples for count batching |
| block-op token buckets | 256, 512, ... up to the largest batch, each run at its top or at the largest batch, whichever is smaller |
| the recorded ops | a weightless twin of the backbone (the same structure, base storage, dtypes and grad flags; no weight memory), on the training device, before `PARALLEL="pipeline"` / `"threaded"` split the model into stages |

- **Text encoder.** The online text encoder's ops are tuned the same way right before
  its first pass, so a run whose captions are all cached tunes none
  ([tuning.md](../internals/tuning.md#prewarm)).
- **Threaded stages.** The kernel ops of the further threaded stage devices bind from
  the same card's tables; the adapter prewarm, which runs on scratch inputs, settles
  every threaded stage device before step 0 (`adapter_prewarm_devices`).
- **Multi-process.** Runs broadcast rank 0's tuning tables afterwards.
- **Adapters.** The adapter set's `prewarm` then settles every distinct adapter
  autotuned op (fused kernel against the compiled formula) over the adapter token
  buckets up to the largest batch (powers of two from 256), on scratch inputs (the
  factors keep their values and no gradient is left) and without calling the layers, so
  streamed layers are covered too and every adapter op is bound before step 0.
- All prewarm runs before the gradient-sync hooks are installed and leaves no gradient
  behind.
- **`TUNE="off"`** binds neither prewarm: every adapter op takes its stored pick, else
  its fused kernel, unmeasured (LyCORIS's own tile tuning is off too). CPU devices are
  skipped (their ops are the untuned eager formula).

After the prewarm the tuner is frozen: a problem key first met later binds its first
candidate untimed, with one WARNING line. Tuning reopens only in these windows
([tuning.md](../internals/tuning.md#tuning-windows)):

- the online text encoder's prewarm, right before its first pass;
- the kernel-vs-compile retune after each of the first `RETUNE_STEPS` steps (default
  3), for the problems queued so far;
- the first sample grid's first sampler step, for its own shapes.

After the last retune step the kernel-vs-compile choice is sealed: no problem is queued
or retuned again. The first sample grid's first sampler step still tunes its own shapes
when it comes later.

`release_workspaces` (gc, device sync, `empty_cache` on every prewarmed device) runs
after the kernel prewarm and again after the adapter prewarm. The first release must
come before the adapter prewarm: the kernel prewarm's freed workspace (512 MiB measured
on an RTX 3090) otherwise takes the adapter prewarm's small persistent allocations and
its segment stays reserved through training.

### Adapter step state

Around each micro-batch the adapter set's `begin_step(sigma, seq, step)` sets T-LoRA
masks and draws DyLoRA ranks / module-dropout skips (seeded by step and micro-batch);
`end_step()` after the step's last backward, and after a validation pass, clears the
masks, so sampling and exports see the full-rank adapter.

### Linear-algebra backend

On a CUDA device each training worker calls
`torch.backends.cuda.preferred_linalg_library("cusolver")` once at start, which sets
the backend for that whole process and run (OFT's batched Cayley inverse; MAGMA's
batched inverse faults on sm86 with the cu130 wheels). Sampling through `MossPipeline`
during training selects cuSOLVER per generate call and restores the previous backend,
which in a training process is this same setting.

### Data and online caches

- `MAX_SAMPLES` caps samples per token batch (0 = none); `TOKEN_BUDGET="exact"` plans
  with the cached text lengths so `MAX_TOKENS` is a hard bound.
- With a cache `online`, a batch whose keys are all cached arrives encoded and is used
  as is. Otherwise it is encoded (latents at the full planned grid, then gathered to
  the token-drop subset; text cut to each sample's `text_limit`) and the full encodings
  go to the data side's background `CacheWriter`, flushed before every checkpoint and
  closed at the end.
- A fresh encoding is rounded exactly as the cache stores it before it is used, so a
  miss trains on the value a later hit returns.
- The batch layout (SeqInfo) is built in the loader worker when the batch's text
  lengths are final (offline text cache or an online hit) and travels with the batch;
  only batches whose prompts are encoded on the device build it on the main process
  (about 0.7 ms per 4-sample, 16k-token batch on CPU).

## Objective

`z_σ = (1-σ) z0 + σ ε`; the model is evaluated at σ and predicts `ε - z0`.

- **Loss.** The per-image-token MSE over the packed batch, normalised by the **global**
  number of image tokens of the optimiser step: all `GRAD_ACC` micro-batches of a step
  are fetched first, their token counts summed (and all-reduced across ranks), and
  every micro-batch loss is divided by that one denominator. Accumulating `GRAD_ACC`
  loader batches therefore gives exactly the gradient of one batch holding all of them.
- **`LOSS`** selects `mse`, `huber` or `l1` (the pipeline engine's last stage uses the
  same loss); `masked_mse` exists in the registry but is rejected by validation, since
  no dataset source produces a loss mask.
- **`WEIGHTING`** multiplies a σ-dependent weight into the data weight (`none` is the
  MOSS recipe; `x0` and `eps` reproduce x0- / noise-space MSE).

### Stream penalties

`STREAM_LOSS = {term: {...}}` adds penalties on the residual stream at every joint
block output. Both terms target the base model's massive residual channels
([massive-channels.md](massive-channels.md)): a pre-norm block's RMSNorm divisor is a
per-token gain, training can grow a few channels whose main role is to set it, and their
peaks bring the stream near the fp16 maximum.

| term | fields (default) | adds |
| --- | --- | --- |
| `concentration` | `weight`, `k` (32) | `weight` × the mean over blocks of the share of the block output's energy (all tokens of the micro-batch) in its `k` largest channels |
| `peak` | `weight`, `cap` (32768), `top` (1e-4), `tau` (0.25), `sigma` ("sigma") | `weight` × the mean over samples, each weighted by `sigma` (σ, 1 − σ or 1), of the mean over blocks of `softplus((log m − log cap) / tau)`; `m` is the mean of the sample's largest `top` fraction of \|x\| in that block's output, in original units |

`CONCENTRATION_LOSS = {"weight": w, "k": k}` is the `concentration` term alone; set one
of the two knobs.

- **Concentration.** Scale-free: it prices concentrating the energy, not its size. Keep
  the weight small (0.01 with `k = 32` on MOSS): a large weight removes the channels and
  also the gain path they provide, which costs task loss.
- **Peak.** A soft cap on the stream's largest values. Below `cap` the hinge is about
  zero; above it, it rises like `(log m − log cap) / tau`. The top fraction spreads the
  gradient over the peak elements instead of one maximum, and it is taken per sample, so
  it holds for every resolution. `m` is divided by the stream scale of the fp16 fix the
  model carries (the weight fold, or `FP16_FIX="online"`'s factors), so `cap` means the
  same with or without a fix (the fp16 maximum is 65504).
- Each micro-batch adds its penalty weighted by its share of the step's loss tokens, so
  the step total is the token-weighted mean, as for the loss itself.
- The training step only: validation reports the plain loss. Logged metrics:
  `concentration`, `peak_loss` (the unweighted hinge) and `stream_peak` (the largest `m`
  of the micro-batch, original units).
- The block outputs are read by forward hooks, so the runner must call whole blocks:
  `CHECKPOINT="sub-block"` is rejected, and the layouts are `PARALLEL` `none` and `ddp`.

## Timestep samplers

| `TIMESTEP` | sampler |
| --- | --- |
| `moss` (default) | the tech-report sampler, below |
| `uniform`, `logit_normal` | continuous alternatives |

`moss` is a discrete grid σ_j = j/1000 with a truncated logit-normal pmf
(`S = 1 - sigmoid(Z)`, `Z ~ N(μ, s²)`, cutoff `a` rescales S/a onto the grid), a uniform
mix `uniform_mix`, mass `p_full` moved to σ = 1, and a static shift. The defaults
(μ = -1.09, s = 0.95, cutoff 0.95) put the median σ at 0.78. `schedule="tech_report"`
switches parameters by step exactly as the four Stage-3 intervals of the report.

With `F(x) = P(S <= x) = 1 - Phi((log((1-x)/x) - mu)/s)`, grid point j gets mass
`(1 - rho) * [F(a j/T) - F(a (j-1)/T)] / F(a) + rho/T` (rho = `uniform_mix`, T = 1000)
before the `p_full` move and the shift map. The table is built in fp32: a bin below the
median is a difference of `F` values, one above it a difference of survival values
`1 - F` (each its own `erfc`), so no bin subtracts two numbers near 1; every bin is
within 1e-6 absolute (3e-4 relative) of the fp64 table.

## Optimisers

| `OPTIMIZER` | what it is |
| --- | --- |
| `adamw` | torch AdamW (fused on CUDA) |
| `adamw16` | AdamW for 16-bit parameters: fp32 math, 16-bit states, `sqrt(v)` stored so the second moment does not underflow in fp16; `OPTIMIZER_KWARGS={"rounding": "nearest" \| "stochastic" \| "kahan"}` (stochastic is bf16-only) |
| `adamw8bit` | bitsandbytes 8-bit AdamW, extra `quant` |
| `adamw8bit_kernel` | KohakuMoss's own 8-bit AdamW: m (int8) and sqrt(v) (uint8) in blockwise states, stepped by the kernel library's `adamw8bit_step` (Triton / TileLang / eager, chosen per device); `OPTIMIZER_KWARGS={"rounding": ...}` as `adamw16` |
| `muon` | Muon on 2-D hidden matrices (fused `qkv` split in 3, `ffn_up_gate` in 2, each chunk orthogonalised by 5 Newton–Schulz steps, update scaled by 0.2·√max(rows, cols) to match AdamW's RMS) and AdamW on the rest, in one optimiser |
| `prodigy`, `schedule_free` | prodigyopt / schedulefree, extra `optim` (schedule-free switches to eval weights for sampling and validation) |

- **`OPTIMIZER` vs `OPTIM_STATE`.** `OPTIMIZER` names the algorithm and its
  implementation; `OPTIM_STATE` (`gpu`, `stream`, `cpu`) only places `adamw`'s states
  and picks its placement builder (`gpu` torch AdamW, `stream` `host_state_adamw`, `cpu`
  `cpu_adamw`, also when weight streaming keeps the trainable masters on the host). The
  placement builders (`cpu_adamw`, `host_state_adamw`, `master_adamw`) are not
  `OPTIMIZER` names; validation names the `OPTIMIZER` / `OPTIM_STATE` pair to use
  instead.
- **`OPTIMIZER_KWARGS`** is checked against the registered builder's own arguments
  (`adamw16`: `rounding`; `muon`: `momentum`, `ns_steps`; `schedule_free`:
  `warmup_steps`; `adamw8bit_kernel`: `rounding`; `adamw`, `adamw8bit`, `prodigy`:
  none). `LR`, `BETAS` and `EPS` have their own knobs, and a dotted-path `OPTIMIZER` is
  not checked.
- **`WD_RULES="moss"`** disables weight decay for biases, norms, 1-D tensors and the
  embedders, and applies decay-to-one (`p -= lr·wd·(p-1)`) to the last joint block's q/k
  norm weights in full finetunes.
- **`SCHEDULE`** is an AnySchedule config; `end: -1` means `STEPS`. Loading a checkpoint
  writes the restored step's scheduled values (lr and any other scheduled key) into the
  optimiser groups the schedule drives, including the lr holders of the per-parameter
  and offload optimisers, so the first resumed step runs at the scheduled lr.

## The optimiser step

`OptimBundle.step()`:

1. unscales the gradients;
2. takes their fp32 global norm (per-tensor norms in fp32 too);
3. checks the norm and the loss for non-finite values and agrees that flag across ranks
   (under DDP / FSDP2 the loss shares and the flags go in one SUM all-reduce);
4. brings loss, norm and flag to the host in **one** transfer: the only host sync of a
   training step (the step's token count reaches the device through pinned memory
   without one).

A bad step is skipped (the fp16 scale backs off by 0.5, floored at 1) and counted
(`skipped_steps` and `nonfinite_steps`); clipping multiplies the gradients by an
on-device coefficient.

### Fused step

With `fused_step` in the memory plan (the `low` / `minimal` profiles) each parameter
steps inside backward on the last micro-batch of a step (`FusedStepBundle`). The full
gradient never exists at once, so:

- **Clipping.** `GRAD_CLIP` (the global norm in every bundle) scales each gradient by
  `min(1, GRAD_CLIP / the previous step's global norm)`, the last global norm known when
  a parameter steps (no clip on a run's first step; a non-finite step keeps the previous
  factor; checkpoints carry it). A gradient spike is therefore applied with the previous
  step's factor on the step it occurs (unclipped when the previous norm was below
  `GRAD_CLIP`); its own norm sets the next step's factor, and a non-finite spike is
  skipped by the non-finite guard.
- **`grad_norm`** is the global norm of the unclipped gradients accumulated on device.
- **Not with Muon.** Muon steps whole matrices after backward, so knob validation
  rejects `OPTIMIZER='muon'` with a `fused_step` plan (`kohakumoss.training.knobs`):
  full finetune with Muon uses a profile without `optfused` / stream (e.g.
  `ckpt-resident`), or `FUSED_STEP=False` with `MEMORY_PROFILE='manual'`.
- **DDP.** Each parameter steps on its gradient summed across ranks (see
  [multi-gpu.md](../guides/multi-gpu.md)).
- **Non-finite parameters.** A parameter whose gradient is non-finite is not updated
  (states untouched); the rest of the step applies. Such a step counts in
  `nonfinite_steps` (the scale backs off, an instability dump is written), not in
  `skipped_steps`, and EMA follows it; its `grad_norm` is NaN and `nonfinite_params`
  counts the parameters it froze.

### Stopping on non-finite steps

In both modes `NONFINITE_PATIENCE` (default 8) stops the run, naming the parameters,
after that many consecutive non-finite steps at the loss-scale floor or that many steps
in a row with the same non-finite parameters. Before it stops, it writes the failing
step to `<output>/instability/repro_step.pt` (`training/callbacks/repro.py`, a
`torch.save` dict):

| key | contents |
| --- | --- |
| `z`, `text`, `sigma`, `target`, `hw`, `text_len` | the step's last prepared micro-batch as host tensors (`z`: noised latents) |
| `weight` | the weighted per-sample loss weights |
| `img_keep` | per sample the kept grid indices or None |
| `denom` | the step's token count, an int |
| `step` | the step |
| `adapters` | every trainable tensor by name, its values after the step (a skipped step leaves them as its forward saw them) |

The error message names the file; `KOHAKUMOSS_REPRO_STEP=<file>` replays it in
`tests/training/checkpoint/test_repro_step_nan.py`. The native and Lightning loops
record it; the pipelines do not.

## Precision

| knob | values |
| --- | --- |
| `PRECISION` | `bf16`, `fp16` or `auto` (bf16 when every training CUDA device has it, else fp16; bf16 off CUDA) |
| `FP16_FIX` | `none`, `online` (scales the residual stream at run time; `FP16_FIX_KWARGS={"c": 1/16}` for the base, whose stream peaks near 62k), `fp32_residual` (carries the stream in fp32); see [numerics.md](numerics.md#fp16_fix-modes) |

- Trainable parameters live in fp32 masters (16-bit for `adamw16`), modules in
  `KEEP_FP32` stay fp32, the forward runs under autocast.
- `fp16` enables the dynamic loss scaler (init 2^16, growth ×2 after 2000 clean steps).
- An fp16-fix checkpoint (the fold baked into the weights) trains with
  `FP16_FIX="none"` and that file as `WEIGHTS`; `FP16_FIX="weights"` is not a training
  value (inference loads either).
- Validation rejects `PRECISION="fp16"` on `VARIANT="base"` / `"sft"` with
  `FP16_FIX="none"`, and `PRECISION="bf16"` when any entry of `DEVICES` has no bf16.

## Memory plan in the loop

- **Optimiser state.** The optimiser follows the plan's `optim_state` for
  `OPTIMIZER="adamw"`: `gpu` torch AdamW, `stream` GPU AdamW with host-resident states
  (`host_state_adamw`), `cpu` host AdamW (`cpu_adamw`; also forced when trainable
  weights stream from host masters).
- **Gradient offload.** With `grad_offload` each gradient leaves the device as soon as
  it is accumulated and is summed in fp32 host memory over the `GRAD_ACC` micro-batches;
  the host AdamW reads those sums directly, a device optimiser gets them back before its
  step, and with `fused_step` the last micro-batch's backward finishes each parameter in
  place.
- Every optimiser path keeps the parameter groups (weight-decay rules, LoRA+ up-side
  lr), and the schedule keeps each group's lr ratio.
- Free cached device memory stays with the allocator between steps unless
  `STEP_RELEASE` asks otherwise ([profiles.md](profiles.md#releasing-cached-memory)).

## Pipeline layouts (Linux)

| `PARALLEL` | layout | `PIPELINE_SCHEDULE` |
| --- | --- | --- |
| `threaded` | the stages on `DEVICES` inside one process | `gpipe` or `async` |
| `pipeline` | one process per stage (torch.distributed pipelining) | the torch schedules |

- A step is `MICROBATCHES` loader batches; every pipeline rank reads the same batches
  and steps only the parameters of its own stages, and checkpoints and the final export
  merge the stages on rank 0.
- Each micro-batch's adapter step state (T-LoRA rank masks, DyLoRA ranks, module-dropout
  draws) is drawn in micro-batch order, as the native loop draws it, and installed again
  before that micro-batch's own forward and backward (its checkpoint recompute
  included); the `pipeline` engine draws it from the padded static layout it runs.
- The threaded `async` schedule runs a micro-batch's backward while the next one's
  forward runs, so those adapters with a checkpointing plan are rejected there (use
  `gpipe` or `checkpoint='none'`).
- `pipeline` stages are built for a static micro-batch of `MAX_TOKENS` tokens and
  `MAX_SAMPLES` samples, so it needs token batching, `MAX_SAMPLES >= 1`,
  `TOKEN_BUDGET="exact"` and an offline text cache; token drop is supported (each
  sample's loss covers its kept tokens).
- Both are rejected on Windows, and sampling / validation callbacks are not available
  in these layouts.

## EMA

`EMA = 0.9999` (or a list of decays) keeps fp32 averages updated after each successful
step.

- **Placement.** Each average lives on its parameter's device unless the spec names a
  `device`. Under `stream='swap'` the block masters are host tensors, so their averages
  are host buffers updated on the host, after the in-backward block optimiser's write-back to those masters has landed
  (`StreamedAdamW.synchronize`); the host update costs CPU time per step and parameter.
- **Memory.** The memory estimate counts 4 B per trainable device parameter per
  average.
- **Conversion.** A parameter that is not fp32 on the averages' device is converted once
  per update, in chunks of at most 64 M elements, and averaged into every decay from
  that one copy; accelerator parameters averaged into host buffers are copied through
  two pinned staging chunks (one copies while the other is averaged).

### Power EMA

`EMA = {"kind": "power", "sigma_rel": [0.05, 0.10]}` keeps EDM2 power-function averages
and writes a snapshot at every checkpoint;

```bash
python -m kohakumoss.training.ema.posthoc --run <dir> --sigma-rel 0.07 --out ema.safetensors
```

reconstructs an average of any width afterwards.

- The power-EMA exponent for a `sigma_rel` is the largest root of
  `g^3 + 7 g^2 + (16 - t) g + 12 - t` (t = `sigma_rel^-2`), found by fp32 Newton from
  `g = sqrt(t)`, where the cubic is convex and increasing.
- The post-hoc fit is the least-squares combination of snapshot profiles
  `p(tau) = (g+1) tau^g / t^(g+1)`, solved in fp32 in the basis of unit-norm profiles:
  their Gram matrix is
  `sqrt((2g_a+1)(2g_b+1))/(g_a+g_b+1) (lo/t_a)^(g_a+1/2) (lo/t_b)^(g_b+1/2)`
  (lo = min(t_a, t_b)), every term O(1), where the raw Gram matrix overflows fp32
  (`t^(g+1)`).

## Drivers, checkpoints, callbacks

### Drivers

`DRIVER="native"` (default) and `"lightning"` (manual optimisation) run the same
TrainStep and bundle, start every micro-batch the same way (fused steps, gradient
offload, the DDP reducer) and reach identical parameters.

Lightning drives the loop on this process's one device; processes, gradient sync and
placement are KohakuMoss's. The LightningModule holds the backbone outside its module
tree (a plain attribute), so it has no parameter of its own: Lightning's move of the
module to the device at fit start and back to the CPU at teardown moves nothing, and
every memory profile (host fp32 masters under `swap`, streamed weights, host optimiser
state, the DDP shared host region) keeps its placement; the export after training reads
the same placed model as under the native driver.

### Checkpoints

A checkpoint directory holds the trainable state, EMA, optimiser + schedule + scaler,
the loader position, and one `rng.<rank>.pt` per rank (global torch / numpy / python
RNG, the CUDA state of that rank's devices, the σ and noise generators).

- **Exact resume.** Each rank restores its own RNG file, so a resumed run draws the same
  data, σ and noise streams as an uninterrupted one. On CPU the resumed parameters are
  bit-exact; on CUDA kernels that accumulate through atomics (the fused attention
  backward, cuBLAS split-K, cuDNN autotune) round in arrival order, so the resumed run
  matches to rounding.
- **Atomic write.** A checkpoint is written into `step_XXXXXXXX.partial/` (every rank its
  RNG file, rank 0 the rest) and renamed once complete; only renamed directories are
  listed or resumed, and the last step is not written again when the cadence already
  saved it.
- **`SAVE_MINUTES`** is checked every `LOG_EVERY` steps, and every rank takes rank 0's
  decision.
- **`FINAL_CHECKPOINT`** sets the resume checkpoint at the run's end: `"auto"` (the
  default) writes it only when `SAVE_EVERY` or `SAVE_MINUTES` is on, `True` / `False`
  force it. The exports (`SAVE_FORMAT`) are written either way; a full finetune's resume
  checkpoint holds the fp32 masters and the optimiser state (about 15 B per trainable
  parameter).

### Callbacks

| callback | output |
| --- | --- |
| throughput / MFU, VRAM peaks | metrics |
| metric logging | console + `metrics.jsonl`, or wandb |
| samples | through `MossPipeline` on the live model: images + `settings.json` per step |
| validation | fixed-noise validation loss per σ (the global per-token mean over every rank's validation batches, all conditional: no text dropout) |
| instability dumps | `instability/step_X.json` on a non-finite step or a loss above 5× its running average |

## Shutdown

`training.shutdown.install` (the first thing `entry.run` does; also the bench matrix's
run wrapper) makes SIGTERM raise `SystemExit(143)` in the main thread, so the run
unwinds like an exception (finally blocks, exit handlers), and at exit kills every live
multiprocessing child and the process's forkserver.

Loader workers on Linux are forked by the forkserver: they watch the forkserver, and the
forkserver outlives a trainer killed outright, so the exit handler kills the forkserver
and its workers (about 0.5 to 0.8 GB RSS each) when a run is terminated. SIGKILL cannot be handled; the bench matrix runner starts each run as its own
process group and kills the group (`kohakumoss_bench/run/proc.py`).
