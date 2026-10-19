# Numerics

What runs in which dtype, which PyTorch precision defaults apply, and how fp16 runs
keep the residual stream in range.

| question | section |
| --- | --- |
| which dtype does each part run in? | [Dtypes](#dtypes), [Conditioning](#conditioning) |
| which `FP16_FIX` for an fp16 run? | [FP16_FIX modes](#fp16_fix-modes) |
| why fp16 needs a fix at all | [Why fp16 needs a fix](#why-fp16-needs-a-fix), [The fold](#the-fold) |

## Dtypes

- The trunk runs in the compute dtype (bf16 or fp16).
- `time_embedding`, `shared_block_modulation` and `video_head_modulation` store fp32
  weights and biases; RoPE bands are always fp32.
- Eager ops accumulate in at least fp32 and round once.

## Conditioning

- **Linears.** The four conditioning linears (`time_embedding.0`, `time_embedding.2`,
  `shared_block_modulation`, `video_head_modulation`) are plain PyTorch linears: each
  takes its input in its weight dtype (fp32 for `KEEP_FP32`) and runs under the
  caller's autocast like any other module, so in training and sampling autocast runs
  them on 16-bit operands and they return what PyTorch returns (the sampler's DiT
  forwards run under training's autocast, `kohakumoss.precision.autocast_for`). The
  outputs are cast to fp32 for the per-sample modulation math.
- **Sinusoid.** `timestep_embedding` is the official MOSS formula, `[cos | sin](t f_k)`
  with `t = 1000 sigma` and `f_k = 10000^(-k/128)`, evaluated in fp32. `t f_k` reaches ~1e3 rad, whose fp32 ulp is 6e-5,
  so the phase carries up to ~1e-4 rad of rounding at large t.

## PyTorch defaults

PyTorch 2.14's precision defaults apply throughout:

| switch | default | effect |
| --- | --- | --- |
| cuDNN fp32 convolution (`torch.backends.cudnn.allow_tf32`) | True | fp32 convolutions run TF32 on sm80+ |
| `allow_fp16_reduced_precision_reduction`, `allow_bf16_reduced_precision_reduction` | True | cuBLAS may reduce split-K partials of a 16-bit GEMM in 16 bit |

Autocast and dtype per part:

| part | autocast | dtype |
| --- | --- | --- |
| DiT, training and sampler | inside | 16-bit GEMM inputs |
| VAE | outside | `VAE_DTYPE`, whose `auto` is bf16 where the card has it (sm80+, Intel Arc) and fp16 or fp32 elsewhere; `VAE_DTYPE="fp32"` convolutions run IEEE fp32 on sm70 / sm75 and TF32 on sm80+ by the default above |
| text encoder | outside | `TEXT_ENCODE["dtype"]`: bf16 or fp16 |

## FP16_FIX modes

`FP16_FIX` in the train and sample scripts takes the run-time modes `none`, `online` and
`fp32_residual`. The `weights` mode is what `scripts/export/fp16_fix.py` writes into an
fp16-fix checkpoint; a run trains or samples that file with `FP16_FIX="none"`, and the
fold comes from its metadata.

| mode | what changes | per-block factors |
| --- | --- | --- |
| `none` | nothing at load or run time; a fold the loaded weights already carry (an fp16-fix checkpoint's `kohakumoss_fp16_fix` metadata) stays applied | — |
| `weights` (the fp16-fix checkpoint) | the fold written into the weights: `video_embedder`, `text_embedder.proj`, text-block `proj` / `ffn_down` (weight and bias) x c; gate_a / gate_f rows of `shared_block_modulation` x c/s; every block's `o` / `ffn_down` x s | no: the gate is shared by all blocks, so c and s are uniform |
| `online` | original weights; the same factors applied at run time: embedder and text-refiner outputs x c_0, each block's gates x c_l/s_l and writer inputs x s_l, the stream rescaled by c_l / c_{l-1} between blocks, exact eps | yes |
| `fp32_residual` | streams kept in fp32, norm outputs cast to the compute dtype | — |

Any mode accepts `clamp` (saturating clamp of the joint stream at the fp16 maximum, with
on-device counters, `clamp_counts(model)`), as a last resort.

### The fp16-fix checkpoint

`scripts/export/fp16_fix.py` writes `weights`-mode tensors to a normal single-file
checkpoint (same keys and shapes, loadable by ComfyUI and HF code) and records c and s in
the safetensors metadata (`kohakumoss_fp16_fix`).

- **Other runtimes** keep the stock eps (1e-6) instead of `1e-6 c^2`; the output
  differs only slightly because the stream rms is far above `sqrt(eps) / c`, and
  `EPS_CHECK = True` measures the difference on the real weights.
- **KohakuMoss** reads the metadata: the model is recorded as folded (savers and adapter
  exports undo it), and `fp16_fix_eps="exact"` switches to the exact eps.
- **Quantized bases** built from the file quantize its stored values
  ([quant-kernels.md](../internals/quant-kernels.md#source)). Written at
  `SAVE_DTYPE="fp16"` (the shipped file), those are the fp16 cast of the folded bf16
  weights, so an `int8` / `fp8` / `fp8_compute` / `mxfp8` base from it holds the codes of
  that cast; only a file written at `SAVE_DTYPE="bf16"` keeps the bf16 values (the fold
  scales are powers of two, exact in bf16).

### Adapters and gradients

With the uniform fold, adapters on `q`, `k`, `v`, `ffn_up_gate` (read through an
RMSNorm) and on `o`, `ffn_down` (written through the gates, s = 1) see identical inputs
and produce identical outputs, so `weights` and `online` give the same function and the
same adapter gradients.

Differences appear only for trained parameters inside the rescaled writers (the
embedders, the text-refiner outputs, the modulation gate rows, `o` / `ffn_down` when
s != 1) and for full finetuning, where Adam's step is relative to a weight that is c
(or s) times smaller. `online` exists for those cases: the weights keep their scale.

## Why fp16 needs a fix

The fp16 probe (`scripts/probe/activations.py`, fp32 model, prompts x sigmas x
resolutions) measures the joint residual stream at up to ~6.2e4, within 6 % of the fp16
maximum 65504, while every other value is small:

| value | max |
| --- | --- |
| joint residual stream | ~6.2e4 |
| attention logits | ~600 |
| `o` outputs | ~1.3e3 |
| `ffn_down` outputs | ~700 |
| `ffn_up_gate` | ~170 |
| text stream | ~1.8e3 |

The stream is the only thing to move.

## The fold

Every reader of a residual stream (joint and text) is an RMSNorm, and RMSNorm is scale
invariant except for its eps: `rms(c x)` with eps `c^2 eps` equals `rms(x)` with eps. So
scaling every writer of a stream by `c` and every reading norm's eps by `c^2` leaves the
function unchanged while the stream shrinks by `c`. A pre-gate scale `s` additionally
moves magnitude from a block's `o` / `ffn_down` output into its gate (`o * s`,
`gate / s`).

### Fold arithmetic

The fold multiplies weights in fp32 (or in the tensor's own dtype when wider) and rounds
once. For a power-of-two c and s the product is exact in every dtype except where a
folded fp16 value lands in the subnormal range.

## Saturation counters

`SaturationCounters(model, threshold)` hooks every block and keeps the running max |x|
and the count of values above `threshold` on the device; `read()` syncs once. A block
under activation checkpointing runs its forward again inside backward; the hooks skip
that recompute (a backward graph task is running), so each value is counted once per
forward.
