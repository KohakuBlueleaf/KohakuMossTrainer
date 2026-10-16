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
every rank) and with `SEED + rank` after it.

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
decides, outside autograd, any raced-op problem first met during it. Every adapted
layer then runs once per LyCORIS token bucket up to the largest batch (powers of two
below LyCORIS's row bucket, multiples of it above) and per grad pattern (input with /
without grad, and grad disabled for sampling and validation), so LyCORIS's tier race
settles before step 0; layers inside weight-streamed blocks race live. All prewarm runs
before the gradient-sync hooks are installed and leaves no gradient behind. `TUNE="off"`
binds none of these actions; CPU devices are skipped (their ops are the untuned eager
spec). On CUDA the linear-algebra backend is cuSOLVER (OFT's Cayley solve).

### Data and online caches

`MAX_SAMPLES` caps samples per token batch (0 = none); `TOKEN_BUDGET="exact"` plans
with the cached text lengths so `MAX_TOKENS` is a hard bound. With a cache `online`, a
batch whose keys are all cached arrives encoded and is used as is; otherwise it is
encoded (latents at the full planned grid, then gathered to the token-drop subset; text
cut to each sample's `text_limit`) and the full encodings go to the data side's
background `CacheWriter`, flushed before every checkpoint and closed at the end.

## Objective

`z_σ = (1-σ) z0 + σ ε`, the model is evaluated at σ and predicts `ε - z0`. The loss is the
per-image-token MSE over the packed batch, normalised by the **global** number of image
tokens of the optimiser step: all `GRAD_ACC` micro-batches of a step are fetched first,
their token counts summed (and all-reduced across ranks), and every micro-batch loss is
divided by that one denominator. Accumulating `GRAD_ACC` loader batches therefore gives
exactly the gradient of one batch holding all of them. `LOSS` selects `mse`, `huber`,
`l1` or `masked_mse`; `WEIGHTING` multiplies a σ-dependent weight into the data weight
(`none` is the MOSS recipe; `x0` and `eps` reproduce x0- / noise-space MSE).

## Timestep samplers

`TIMESTEP="moss"` is the tech-report sampler: a discrete grid σ_j = j/1000 with a
truncated logit-normal pmf (`S = 1 - sigmoid(Z)`, `Z ~ N(μ, s²)`, cutoff `a` rescales
S/a onto the grid), a uniform mix `uniform_mix`, mass `p_full` moved to σ = 1, and a
static shift. `schedule="tech_report"` switches parameters by step exactly as the four
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
A bad step is skipped (the fp16 scale backs off by 0.5, floored at 1) and counted;
clipping multiplies the gradients by an on-device coefficient. With `fused_step` in the
memory plan (the `low` / `minimal` profiles) each parameter steps inside backward on the
last micro-batch of a step (`FusedStepBundle`): the full gradient never exists at once,
so `GRAD_CLIP` clips each parameter by its own norm, and the mode is single-device and
not combined with Muon. The optimiser state follows `optim_state` of the plan: `adamw`
becomes `adamw8bit` for `8bit` and `cpu_adamw` for `cpu`.

## Precision

`bf16` / `fp16` / `fp32` compute; trainable parameters live in fp32 masters (16-bit for
`adamw16`), modules in `KEEP_FP32` stay fp32, the forward runs under autocast. `fp16`
enables the dynamic loss scaler (init 2^16, growth ×2 after 2000 clean steps).
`FP16_FIX` keeps the residual stream inside the fp16 range: `online` scales it at run
time (`FP16_FIX_KWARGS={"c": 1/16}` for the base, whose stream peaks near 62k),
`weights` bakes the same fold into the weights (adapter and merged exports undo it),
`fp32_residual` carries the stream in fp32 (see numerics).

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
`PIPELINE_SCHEDULE`). A step is `MICROBATCHES` loader batches; every pipeline rank reads
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
reconstructs an average of any width afterwards.

## Drivers, checkpoints, callbacks

`DRIVER="native"` (default) and `"lightning"` (manual optimisation) run the same TrainStep
and bundle and reach bit-identical parameters. A checkpoint directory holds the trainable
state, EMA, optimiser + schedule + scaler, every RNG (global, σ sampler, noise) and the
loader position, so resuming is bit-exact. Callbacks: throughput / MFU, VRAM peaks,
metric logging (console + `metrics.jsonl`, KohakuBoard or wandb), samples through
`MossPipeline` on the live model (images + `settings.json` per step), fixed-noise
validation loss per σ, and instability dumps (`instability/step_X.json` on a skipped step
or a loss above 5× its running average).
