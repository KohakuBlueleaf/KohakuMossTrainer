# Numerics

## Dtype policy

The trunk runs in the compute dtype (bf16, fp16 or fp32). `time_embedding`,
`shared_block_modulation` and `video_head_modulation` store fp32 weights and biases; RoPE
bands are always fp32. Eager ops accumulate in at least fp32 and round once. No model,
kernel, trainer or inference code computes in fp64; fp64 is the precision reference only
(`tests/rules/test_no_fp64.py`).

## Conditioning

- **Linears.** With a 16-bit trunk (or `backbone.cond_dtype` set), the four conditioning
  linears (`time_embedding.0`, `time_embedding.2`, `shared_block_modulation`,
  `video_head_modulation`) cast input and weight to the 16-bit dtype, multiply with fp32
  accumulation into an fp32 output, and add the fp32 bias (`MossLinear.set_input_dtype`,
  bound by `bind_ops`; re-bind after a dtype change). The modulation leaves the GEMM
  unrounded, so the only 16-bit roundings are the operands, as in every other matmul.
  The GEMM is `torch.mm(..., out_dtype=float32)` where the backend has it (CUDA fp16;
  CUDA bf16 on sm80+). Elsewhere (CPU, bf16 on sm70 / sm75) it is an fp32 GEMM of the
  16-bit-rounded operands: their products are exact in fp32 and the sums are fp32, so
  it computes the same thing. The extra weight cast reads the fp32 weight once per
  forward (~0.35 GB at base size), small against a forward's trunk.
- **Sinusoid.** `cos / sin(t f_k)` for t up to 1000 and `f_k = 10000^(-k/128)` loses
  ~1e-4 rad in a plain fp32 product (`t f_k` is ~1e3 rad, its fp32 ulp is 6e-5). The fp32
  sinusoid instead works in turns, `t f_k / 2 pi`: the host splits each `f_k / 2 pi`
  into a 10-bit head and an fp32 tail, `t` splits into a 14-bit head (multiple of 2^-4)
  and an exact tail, so `t_hi g_hi` is exact in fp32 and its integer part drops
  exactly; the small remaining terms are added in fp32 and scaled by 2 pi only after
  reduction to |turns| < 1. The phase error is ~3e-7 rad (`tests/model/test_conditioning.py`
  bounds the output at 1e-6 against fp64), and fused multiply-adds only make the exact
  products more exact.

## Fold arithmetic

The fold multiplies weights in fp32 (or in the tensor's own dtype when wider) and rounds
once. For a power-of-two c and s the product is exact in every dtype except where a
folded fp16 value lands in the subnormal range (`tests/model/test_fold_exact.py`).

## Why fp16 needs a fix

The fp16 probe (`scripts/probe/activations.py`, fp32 model, prompts x sigmas x
resolutions) measures the joint residual stream at up to ~6.2e4 — within 6 % of the fp16
maximum 65504 — while every other value is small: attention logits <= ~600, `o` outputs
<= ~1.3e3, `ffn_down` outputs <= ~700, `ffn_up_gate` <= ~170, text stream <= ~1.8e3. The
stream is the only thing to move.

## The fold

Every reader of a residual stream (joint and text) is an RMSNorm, and RMSNorm is scale
invariant except for its eps: `rms(c x)` with eps `c^2 eps` equals `rms(x)` with eps. So
scaling every writer of a stream by `c` and every reading norm's eps by `c^2` leaves the
function unchanged while the stream shrinks by `c`. A pre-gate scale `s` additionally
moves magnitude from a block's `o` / `ffn_down` output into its gate (`o * s`,
`gate / s`).

## FP16_FIX modes

| mode | what changes | per-block factors |
| --- | --- | --- |
| `none` | nothing | — |
| `weights` | the fold written into the weights: `video_embedder`, `text_embedder.proj`, text-block `proj` / `ffn_down` (weight and bias) x c; gate_a / gate_f rows of `shared_block_modulation` x c/s; every block's `o` / `ffn_down` x s | no: the gate is shared by all blocks, so c and s are uniform |
| `online` | original weights; the same factors applied at run time: embedder and text-refiner outputs x c_0, each block's gates x c_l/s_l and writer inputs x s_l, the stream rescaled by c_l / c_{l-1} between blocks, exact eps | yes |
| `fp32_residual` | streams kept in fp32, norm outputs cast to the compute dtype | — |

Any mode accepts `clamp` (saturating clamp of the joint stream at the fp16 maximum, with
on-device counters, `clamp_counts(model)`), as a last resort.

### The fp16-fix checkpoint

`scripts/export/fp16_fix.py` writes `weights`-mode tensors to a normal single-file
checkpoint (same keys and shapes, loadable by ComfyUI and HF code) and records c and s in
the safetensors metadata (`kohakumoss_fp16_fix`). Other runtimes keep the stock eps
(1e-6) instead of `1e-6 c^2`; this is "nearly same" because the stream rms is far above
`sqrt(eps) / c` — `--eps-check` measures the difference on the real weights. KohakuMoss
reads the metadata: the model is recorded as folded (savers and adapter exports undo it),
and `fp16_fix_eps="exact"` switches to the exact eps.

### Adapters and gradients

With the uniform fold, adapters on `q`, `k`, `v`, `ffn_up_gate` (read through an
RMSNorm) and on `o`, `ffn_down` (written through the gates, s = 1) see identical inputs and
produce identical outputs, so `weights` and `online` give the same function and the same
adapter gradients (`tests/model/test_fp16fix.py` checks LoRA gradients on every block
linear). Differences appear only for trained parameters inside the rescaled writers —
the embedders, the text-refiner outputs, the modulation gate rows, `o` / `ffn_down` when
s != 1 — and for full finetuning, where Adam's step is relative to a weight that is c
(or s) times smaller. `online` exists for those cases: the weights keep their scale.

## Saturation counters

`SaturationCounters(model, threshold)` hooks every block and keeps the running max |x| and
the count of values above `threshold` on the device; `read()` syncs once.
