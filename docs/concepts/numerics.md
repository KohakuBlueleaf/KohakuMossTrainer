# Numerics

## Dtype policy

The trunk runs in the compute dtype (bf16, fp16 or fp32). `time_embedding`,
`shared_block_modulation` and `video_head_modulation` stay fp32 and run on fp32 inputs;
RoPE bands are always fp32. Each module's tensors are cast once (kept modules never pass
through the compute dtype). Eager ops accumulate in at least fp32 and round once.

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
