# Training

`scripts/train/moss.py` declares every knob as an UPPER_CASE global;
`kohakumoss.training.entry.run(globals())` validates all of them at once (one
`SystemExit` listing every problem, with "did you mean" hints for typos), then builds
in a fixed order and hands the loop to a driver.

## Build order

1. Precision policy (`PRECISION`, `KEEP_FP32`, `FP16_FIX`, `FP16_FIX_KWARGS`).
2. Backbone: `WEIGHTS="random"` builds a random preset (tiny for smoke runs), otherwise
   the checkpoint is loaded straight to the device.
3. `FP16_FIX` (`apply_fp16_fix`), then the memory plan (quantized base, weight
   streaming, activation offload, block runner).
4. Trainable set: LyCORIS adapters (`kohakumoss.adapters`), or `FullTrainable` for
   `ALGO="full"` (the backbone's own parameters; the text refiner stays frozen unless
   `TRAIN_TEXT_REFINER`). Trainable parameters move to the master dtype.
5. `bind_ops()`, encoders, parallel hooks / pipeline stages, kernel prewarm, compile,
   TrainStep, OptimBundle, EMA, data, logger, callbacks, checkpoints, resume, driver.

Everything is resolved here; the loop has no per-step configuration branches. The
global RNG is seeded with `SEED` for the build (identical model and adapter init on
every rank) and with `SEED + rank` after it. The σ and noise generators of the train
step are seeded from `(SEED, stream rank)`: each data-parallel rank draws its own σ and
noise; under `PARALLEL="pipeline"` every stage uses rank 0's streams (all stages see the
same micro-batches).

### Kernel prewarm

Before step 0 the backbone's resolved `attention_varlen` and block ops (norms, RoPE,
SiTU, gated residual) run forward + backward on scratch inputs over every length bucket
the planned batches reach, outside any checkpoint, `no_grad` or inference mode, so no
kernel choice is made inside a checkpointed forward. The buckets come from the data
knobs: the longest sample is `min(MAX_TOKENS, largest planned latent grid +
max_text_tokens)` (attention segment lengths up to it), the largest batch is
`MAX_TOKENS` for token batching or `BATCH_SIZE` samples for count batching (block-op
token buckets 256, 512, ... up to it). `PARALLEL="threaded"` prewarms every stage
device; multi-process runs broadcast rank 0's tuning tables afterwards. Each step then
decides, outside autograd, any raced-op problem first met during it. The adapter set's
`prewarm` then races every adapter tier slot over the LyCORIS token buckets up to the
largest batch (powers of two below LyCORIS's row bucket, multiples of it above), on
random scratch values of the factors (restored afterwards, no gradient left) and
without calling the layers, so streamed layers are covered too and LyCORIS's tier race
settles before step 0. All prewarm runs before the gradient-sync hooks are installed
and leaves no gradient behind. `TUNE="off"` binds neither prewarm and turns LyCORIS's
tier race off (`apply` calls `lycoris.kernels.set_tuning(False)`), so adapter ops take
the first capable tier unmeasured; CPU devices are skipped (their ops are the untuned
eager spec).

Around each micro-batch the adapter set's `begin_step(sigma, seq, step)` sets T-LoRA
masks and draws DyLoRA ranks / module-dropout skips (seeded by step and micro-batch);
`end_step()` after the step's last backward, and after a validation pass, clears the
masks, so sampling and exports see the full-rank adapter.

On a CUDA device each training worker calls
`torch.backends.cuda.preferred_linalg_library("cusolver")` once at start, which sets
the backend for that whole process and run (OFT's batched Cayley inverse; MAGMA's
batched inverse faults on sm86 with the cu130 wheels). Sampling through `MossPipeline`
during training selects cuSOLVER per generate call and restores the previous backend,
which in a training process is this same setting.

### Data and online caches

`MAX_SAMPLES` caps samples per token batch (0 = none); `TOKEN_BUDGET="exact"` plans
with the cached text lengths so `MAX_TOKENS` is a hard bound. With a cache `online`, a
batch whose keys are all cached arrives encoded and is used as is; otherwise it is
encoded (latents at the full planned grid, then gathered to the token-drop subset; text
cut to each sample's `text_limit`) and the full encodings go to the data side's
background `CacheWriter`, flushed before every checkpoint and closed at the end. A
fresh encoding is rounded exactly as the cache stores it before it is used, so a miss
trains on the value a later hit returns.

The batch layout (SeqInfo) is built in the loader worker when the batch's text lengths
are final (offline text cache or an online hit) and travels with the batch; only
batches whose prompts are encoded on the device build it on the main process (about
0.7 ms per 4-sample, 16k-token batch on CPU).

## Objective

`z_σ = (1-σ) z0 + σ ε`, the model is evaluated at σ and predicts `ε - z0`. The loss is the
per-image-token MSE over the packed batch, normalised by the **global** number of image
tokens of the optimiser step: all `GRAD_ACC` micro-batches of a step are fetched first,
their token counts summed (and all-reduced across ranks), and every micro-batch loss is
divided by that one denominator. Accumulating `GRAD_ACC` loader batches therefore gives
exactly the gradient of one batch holding all of them. `LOSS` selects `mse`, `huber` or
`l1` (the pipeline engine's last stage uses the same loss); `masked_mse` exists in the
registry but is rejected by validation, since no dataset source produces a loss mask.
`WEIGHTING` multiplies a σ-dependent weight into the data weight (`none` is the MOSS
recipe; `x0` and `eps` reproduce x0- / noise-space MSE).

## Timestep samplers

`TIMESTEP="moss"` is the tech-report sampler: a discrete grid σ_j = j/1000 with a
truncated logit-normal pmf (`S = 1 - sigmoid(Z)`, `Z ~ N(μ, s²)`, cutoff `a` rescales
S/a onto the grid), a uniform mix `uniform_mix`, mass `p_full` moved to σ = 1, and a
static shift. With `F(x) = P(S <= x) = 1 - Phi((log((1-x)/x) - mu)/s)`, grid point j
gets mass `(1 - rho) * [F(a j/T) - F(a (j-1)/T)] / F(a) + rho/T` (rho = `uniform_mix`,
T = 1000) before the `p_full` move and the shift map. The table is built in fp32: a
bin below the median is a difference of `F` values, one above it a difference of
survival values `1 - F` (each its own `erfc`), so no bin subtracts two numbers near 1;
every bin is within 1e-6 absolute (3e-4 relative) of the fp64 table.
`schedule="tech_report"` switches parameters by step exactly as the four
Stage-3 intervals of the report. The defaults (μ = -1.09, s = 0.95, cutoff 0.95) put the
median σ at 0.78. `uniform` and `logit_normal` are continuous alternatives.

## Optimisers

| `OPTIMIZER` | what it is |
| --- | --- |
| `adamw` | torch AdamW (fused on CUDA) |
| `adamw16` | AdamW for 16-bit parameters: fp32 math, 16-bit states, `sqrt(v)` stored so the second moment does not underflow in fp16; `OPTIMIZER_KWARGS={"rounding": "nearest" \| "stochastic" \| "kahan"}` (stochastic is bf16-only) |
| `adamw8bit` | bitsandbytes 8-bit AdamW |
| `muon` | Muon on 2-D hidden matrices (fused `qkv` split in 3, `ffn_up_gate` in 2, each chunk orthogonalised by 5 Newton–Schulz steps, update scaled by 0.2·√max(rows, cols) to match AdamW's RMS) and AdamW on the rest, in one optimiser |
| `prodigy`, `schedule_free` | prodigyopt / schedulefree (schedule-free switches to eval weights for sampling and validation) |
| `cpu_adamw` | host AdamW; chosen automatically for `adamw` when weight streaming keeps trainable masters on the host |

`WD_RULES="moss"` disables weight decay for biases, norms, 1-D tensors and the
embedders, and applies decay-to-one (`p -= lr·wd·(p-1)`) to the last joint block's q/k
norm weights in full finetunes. `SCHEDULE` is an AnySchedule config; `end: -1` means
`STEPS`.

## The optimiser step

`OptimBundle.step()` unscales the gradients, takes their fp32 global norm, checks the norm
and the loss for non-finite values, agrees that flag across ranks (MAX), and brings loss,
norm and flag to the host in **one** transfer — the only host sync of a training step.
A bad step is skipped (the fp16 scale backs off by 0.5, floored at 1) and counted
(`skipped_steps` and `nonfinite_steps`); clipping multiplies the gradients by an
on-device coefficient. With `fused_step` in the memory plan (the `low` / `minimal`
profiles) each parameter steps inside backward on the last micro-batch of a step
(`FusedStepBundle`): the full gradient never exists at once, so `GRAD_CLIP` clips each
parameter by its own norm, `grad_norm` is the norm of the unclipped per-parameter
gradients accumulated on device, and the mode is single-device and not combined with
Muon. There a parameter whose gradient is non-finite is not updated (states untouched);
the rest of the step applies. Such a step counts in `nonfinite_steps` (the scale backs
off, an instability dump is written), not in `skipped_steps`, and EMA follows it.
The optimiser state follows `optim_state` of the plan: `adamw` becomes `adamw8bit` for
`8bit` and `cpu_adamw` for `cpu`.

## Precision

`bf16` / `fp16` compute (`auto`: bf16 when every training device has it, else fp16;
fp32 on CPU). Every matmul takes 16-bit inputs with fp32 accumulation; fp32 holds the
vector ops, reductions and masters only, so `PRECISION="fp32"` (fp32 matmuls) is the
CPU / test reference mode and validation rejects it on a GPU. No part of the trainer
computes in fp64. Trainable parameters live in fp32 masters (16-bit for
`adamw16`), modules in `KEEP_FP32` stay fp32, the forward runs under autocast. `fp16`
enables the dynamic loss scaler (init 2^16, growth ×2 after 2000 clean steps).
`FP16_FIX` keeps the residual stream inside the fp16 range: `online` scales it at run
time (`FP16_FIX_KWARGS={"c": 1/16}` for the base, whose stream peaks near 62k),
`weights` bakes the same fold into the weights (adapter and merged exports undo it),
`fp32_residual` carries the stream in fp32 (see numerics). Validation rejects
`PRECISION="fp16"` on `VARIANT="base"` / `"sft"` with `FP16_FIX="none"`, and
`PRECISION="bf16"` when any entry of `DEVICES` has no bf16.

## Memory plan in the loop

The optimiser follows the plan's `optim_state` for `OPTIMIZER="adamw"`: `gpu` torch
AdamW, `8bit` bitsandbytes, `stream` GPU AdamW with host-resident states, `cpu` host
AdamW (also forced when trainable weights stream from host masters). With
`grad_offload` each gradient leaves the device as soon as it is accumulated and is summed
in fp32 host memory over the `GRAD_ACC` micro-batches; the host AdamW reads those sums
directly, a device optimiser gets them back before its step, and with `fused_step` the
last micro-batch's backward finishes each parameter in place. Every optimiser path keeps
the parameter groups (weight-decay rules, LoRA+ up-side lr), and the schedule keeps each
group's lr ratio.

## Pipeline layouts (Linux)

`PARALLEL="threaded"` runs the stages on `DEVICES` inside one process;
`PARALLEL="pipeline"` runs one process per stage (torch.distributed pipelining,
`PIPELINE_SCHEDULE`; `threaded` runs `gpipe` or `async`). A step is `MICROBATCHES` loader batches; every pipeline rank reads
the same batches and steps only the parameters of its own stages, and checkpoints merge
the stages on rank 0. `pipeline` stages are built for a static micro-batch of
`MAX_TOKENS` tokens and `MAX_SAMPLES` samples, so it needs token batching,
`MAX_SAMPLES >= 1`, `TOKEN_BUDGET="exact"` and an offline text cache; token drop is
supported (each sample's loss covers its kept tokens). Both are rejected on Windows,
and sampling / validation callbacks are not available in these layouts.

## EMA

`EMA = 0.9999` (or a list of decays) keeps fp32 averages updated after each successful
step. `EMA = {"kind": "power", "sigma_rel": [0.05, 0.10]}` keeps EDM2 power-function
averages and writes a snapshot at every checkpoint; `python -m
kohakumoss.training.ema.posthoc --run <dir> --sigma-rel 0.07 --out ema.safetensors`
reconstructs an average of any width afterwards. The power-EMA exponent for a
`sigma_rel` is the largest root of `g^3 + 7 g^2 + (16 - t) g + 12 - t` (t =
`sigma_rel^-2`), found by fp32 Newton from `g = sqrt(t)`, where the cubic is convex and
increasing. The post-hoc fit is the least-squares combination of snapshot profiles
`p(tau) = (g+1) tau^g / t^(g+1)`, solved in fp32 in the basis of unit-norm profiles: their
Gram matrix is `sqrt((2g_a+1)(2g_b+1))/(g_a+g_b+1) (lo/t_a)^(g_a+1/2) (lo/t_b)^(g_b+1/2)`
(lo = min(t_a, t_b)), every term O(1), where the raw Gram matrix overflows fp32
(`t^(g+1)`).

## Drivers, checkpoints, callbacks

`DRIVER="native"` (default) and `"lightning"` (manual optimisation) run the same TrainStep
and bundle, arm every micro-batch the same way (fused steps, gradient offload, the DDP
reducer) and reach identical parameters. Lightning drives the loop on this process's one
device; processes and gradient sync are KohakuMoss's.

A checkpoint directory holds the trainable state, EMA, optimiser + schedule + scaler,
the loader position, and one `rng.<rank>.pt` per rank (global torch / numpy / python
RNG, the CUDA state of that rank's devices, the σ and noise generators); each rank
restores its own file, so resuming is bit-exact on every rank: always on CPU, and on
CUDA in deterministic mode. `KOHAKUMOSS_DETERMINISTIC=1`, set before the process starts,
selects the kernels' deterministic backward paths (no atomics), and the trainer then
turns on torch's deterministic algorithms, a fixed cuBLAS workspace
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`) and no cuDNN autotune; kernel tuning choices come
from the per-card tuning table, the same for both runs. Deterministic mode exists for
reproducibility checks (the exact-resume tests) and costs speed (no atomics, no
autotuned cuDNN, deterministic torch kernels); training and every benchmark run without
it, and it is on only when the variable is set explicitly. A checkpoint is written into
`step_XXXXXXXX.partial/` (every rank its RNG file, rank 0 the rest) and renamed once
complete; only renamed directories are listed or resumed, and the last step is not
written again when the cadence already saved it. `SAVE_MINUTES` is checked every
`LOG_EVERY` steps, and every rank takes rank 0's decision.

Callbacks: throughput / MFU, VRAM peaks, metric logging (console + `metrics.jsonl`,
KohakuBoard or wandb), samples through `MossPipeline` on the live model (images +
`settings.json` per step), fixed-noise validation loss per σ (the global per-token mean
over every rank's validation batches, all conditional: no text dropout), and
instability dumps (`instability/step_X.json` on a non-finite step or a loss above 5×
its running average).
