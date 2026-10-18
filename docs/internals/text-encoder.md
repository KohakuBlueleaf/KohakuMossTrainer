# Text encoder: kernels, batching, placement

MOSS-Image conditions on the last hidden states of Qwen3.5-9B (32 decoder layers: 24
Gated DeltaNet layers, 8 causal GQA full-attention layers, hidden 4096). The encoder is
frozen; everything here is forward-only under `torch.no_grad`.

## Ops

`kohakumoss_kernels.ops.TEXT` (signatures in `kohakumoss_kernels/ops.py`, eager
reference in `backends/eager/textenc.py`):

| op | eager reference | kernels |
| --- | --- | --- |
| `causal_conv_varlen` | width-4 depthwise conv as shifted fp32 multiply-adds, taps before the token's segment start masked, silu, one rounding | `compile` (torch.compile of the eager reference) |
| `gated_delta_rule_varlen` | gates, l2 norms and the chunked delta rule (chunk 64) on per-segment padded tensors, plain PyTorch matmuls on input-dtype operands | Triton (sm80+, XPU), TileLang (sm70+), each timed against the eager reference |
| `rms_norm_gated` | `x / rms(x) * w * silu(z)`, fp32, one rounding | `compile` |
| `attention_varlen_causal` | per-segment SDPA (`is_causal`, kv heads repeated) | Triton (sm80+, XPU), TileLang (sm70+), each timed against torch `varlen_attn` (flash, causal window, GQA) where the torch build has flash attention and the card is sm80+, else against the per-segment SDPA eager reference (Windows wheels ship without flash attention) |

The block RMSNorms use the existing `rms_norm` op with the weight `1 + w` (Qwen's
zero-centred norm); the MLP and projections are the model's own linears.

### Gated delta rule designs (Triton)

Per value head h (key head `h // 2`): `g = -exp(A_log) softplus(a + dt_bias)`,
`beta = sigmoid(b)`, `q, k` l2-normalised (q also by `Dk^-1/2`), and per token
`S <- e^g S + beta k (v - e^g S^T k)^T`, `o = S^T q`, S zero at every segment start.

- `recurrent`: one program per (segment, value head, value tile), one token per step,
  fp32 state in registers.
- `chunk`: chunks of 64 tokens. `prep_kernel` builds the UT transform
  `T = (I + A)^-1` (A the strictly lower `beta_i k_i.k_j e^(G_i - G_j)`) by 16x16
  diagonal inverses and block forward substitution. `state_kernel` walks each
  segment's chunks with the state, writing every chunk's start state H and
  `V' = T (beta v - beta e^G k_n H)`; the first chunk runs before the loop (no state
  product, no H) and each loop step folds the previous chunk into the state first, so
  the last chunk's end state is never formed. `out_kernel` (parallel over chunks) writes
  `o = (q_n e^G) H + (q_n k_n^T . decay) V'`.
- `fused`: `prep_kernel` also writes the decayed QK matrix and per-row factors; one
  serial kernel per (segment, head, value tile) makes each chunk's output in the same
  pass (no H / V' round trip). A segment's first chunk skips the state products.
- `short`: segments of at most 64 tokens are packed, whole, several to a 64-row tile;
  the state is zero at each segment start, so a tile's output is in-tile only
  (`o = (QK . same-segment) T (beta v)`, A masked to same-segment pairs). Longer
  segments in the same call go through `chunk` or `recurrent`. The tile plan is built
  on the host once per `cu_seqlens` tensor (one sync per encoder pass; every layer
  reuses it).

The registry entry times every design's tile / warp / precision candidates per
problem key (dtype, heads, head dims, segment count, longest and mean segment buckets)
and binds the winner (`scripts/bench/textenc/ops.py` writes the full timing table).

### Speed

Registry-selected op, ms per call, Qwen3.5-9B head layout (delta: 16 key / 32 value
heads of 128; attention: 16 query / 4 kv heads of 256), bf16, shape = segments x
tokens; `fla` = flash-linear-attention 0.5.2 `chunk_gated_delta_rule` (varlen), `hf` =
transformers' `torch_chunk_gated_delta_rule` (padded batch):

| shape | delta, RTX 3090 | fla | hf | delta, RTX 5090 | fla | hf |
| --- | --- | --- | --- | --- | --- | --- |
| 1x300 | 0.175 | 1.251 | 2.28 | 0.183 | 1.609 | 2.44 |
| 16x300 | 0.893 | 1.669 | 12.45 | 0.333 | 1.624 | 4.34 |
| 64x150 | 1.930 | 3.017 | 28.68 | 0.653 | 1.732 | 11.31 |
| 256x40 | 1.853 | 3.466 | 35.80 | 0.381 | 1.834 | 14.44 |
| 1x2048 | 0.511 | 1.227 | 9.49 | 0.208 | 1.618 | 10.07 |
| 4x1024 | 0.806 | 1.529 | 10.14 | 0.290 | 1.633 | 5.75 |

| shape | attention, RTX 3090 | per-segment SDPA | attention, RTX 5090 | per-segment SDPA |
| --- | --- | --- | --- | --- |
| 1x300 | 0.051 | 0.194 | 0.068 | 0.178 |
| 16x300 | 0.327 | 2.011 | 0.121 | 2.059 |
| 64x150 | 0.403 | 7.730 | 0.155 | 7.674 |
| 256x40 | 0.348 | 28.48 | 0.142 | 28.24 |
| 1x2048 | 0.635 | 0.794 | 0.232 | 0.359 |
| 4x1024 | 0.647 | 1.102 | 0.208 | 0.605 |

In fp16 on the 3090 the delta rule is within 3% of the bf16 column and attention within
0.012 ms.

### Precision

Every matmul takes 16-bit (input-dtype) operands with fp32 accumulation. The state and
every scale and decay stay fp32; T, the chunk start states H, V' and the decayed QK are
only ever MMA operands, so they are stored in the input dtype (that store is the
operand rounding). There is no fp64 and no split (hi / lo) product anywhere in the
kernels. The eager reference is plain PyTorch: the same algorithm with its matmuls on
input-dtype operands, each output in the dtype PyTorch gives (the input dtype). The
error statistics (`kohakumoss_kernels.error_stats`: cosine and abs / rel error vs fp64
next to the eager reference's in the same dtype; a result fails only on a regression)
compare the kernels against that PyTorch result.
`recurrent` is vector math only (no MMA) and is checked the same way. Two choices keep
the chunked kernels at the eager reference's error:

- Decays are computed without prefix-sum differences: within a 16-row block as a
  masked cumulative sum, across blocks as suffix + block totals + prefix, all sums of
  same-sign terms. `G_i - G_j` of two long prefix sums loses the small difference in
  the rounding of the large sums.
- softplus is `log1p(exp(x))` (libdevice); `log(1 + exp(x))` loses most digits of
  small softplus values, and the error accumulates along the decay.

fp32 inputs time only `recurrent` (MMAs take 16-bit operands only).

## Forward and layouts

`kohakumoss.encoders.text.QwenTextEncoder` drops the vision tower, tokenizes with the
training chat template, groups the prompts (longest first, at most `batch` prompts and
`max_tokens` tokens per kernel call) and runs one pass:

| layout | forward | what a call sees |
| --- | --- | --- |
| `packed` | `kernels` | prompts concatenated, `cu_seqlens` boundaries, positions reset per prompt |
| `padded` | `kernels` / `hf` | rows right-padded to the longest (`kernels`: each row a segment of the padded width; `hf`: attention mask) |
| `single` | `kernels` / `hf` | one prompt |

The kernel forward (`encoders/text/forward.py`) runs the transformers modules' own
linears, norms' weights and rotary tables, with the text ops above; each decoder
layer's block function is bound once from its layer type. It runs layer-major over all
packs of a pass. Right padding is exact for real tokens because every layer is causal
(causal attention, causal conv, forward recurrence).

Encode throughput (`scripts/bench/textenc/textenc.py`, RTX 5090, bf16, 64 captions of
20-160 words = 13.2k tokens per pass; tokens/s, peak device GiB during the pass;
`offload` / `stream` move 13.8 GB of weights per pass and hold nothing between passes):

| forward | memory | layout | batch 1 | batch 8 | batch 32 |
| --- | --- | --- | --- | --- | --- |
| kernels | resident | packed | 4671 / 15.1 | 12540 / 15.3 | 14332 / 16.1 |
| kernels | resident | padded | 4713 / 15.0 | 12137 / 15.3 | 10165 / 16.5 |
| kernels | offload | packed | 3890 / 13.1 | 8797 / 13.4 | 9183 / 14.2 |
| kernels | stream | packed | 4652 / 1.8 | 12778 / 2.1 | 12714 / 2.9 |
| kernels | stream | padded | 4554 / 1.8 | 11761 / 2.1 | 9652 / 3.3 |
| transformers | resident | padded | 2383 / 15.0 | 9325 / 15.6 | 6709 / 17.9 |

`stream` runs at the resident speed from batch 8 on: the layer ring hides the weight
copies behind the compute of a batched pass.

Against transformers' own encoder the check is the error-statistics check
(`tests/_util/error_stats.py`) with transformers' encode at the same precision as the
baseline, in the same batching (its unbatched encode for one prompt per call, its
padded batch for the batched layouts): the stats table against the fp64 encode is
recorded, and a layout fails only on a regression. The fp64 encode is the kernel forward on the host, whose ops
are checked against independent references; transformers' own forward upcasts its
delta rule and norms to fp32 and so is no fp64 reference.

On real weights (32 captions, 1-300 tokens; `tests/model/test_text_encoder_real.py`,
RTX 5090), abs error / rel error:

| dtype | transformers single | transformers padded | kernels single | kernels packed | kernels padded | packed, one merged segment |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 22.7 / 3.1e5 | 21.1 / 1.2e6 | 10.5 / 6.8e5 | 16.8 / 6.2e5 | 22.5 / 1.0e6 | 76.6 / 4.1e6 |
| fp16 | 1.78 / 1.5e4 | 3.54 / 6.9e4 | 2.07 / 1.1e5 | 7.89 / 1.5e4 | 2.13 / 3.9e4 | |

The rel error is set by near-zero reference elements, the abs error by the few
prompts whose rounding differences grow through the 32 layers; either moves by a factor
of about 2 with any change of GEMM shape or reduction order (transformers' own two
batchings differ by 2x in fp16 abs and 4x in rel). The packed pass with one merged
segment (prompts attending and recurring across their boundaries) exceeds both bounds
(3.4x abs, 13x rel of transformers' unbatched encode).

## TextEncoderPlan

One knob set (`kohakumoss.types.plans.TextEncoderPlan`) for training (`TEXT_ENCODE`,
`TEXT_DEVICE`), the pipeline (`text_plan=`, `sample.py --text-*`) and the offline cache
factory (`kohakumoss.encoders.text.text_encoder`): `memory`, `free_after_use`,
`precision` / `quant_group`, `dtype`, `stream_resident`, `pinned`, `batch`,
`max_tokens`, `layout`, `forward`, `ahead`, `ahead_device`, `kernels`. Placements are
in [offload.md](offload.md#text-encoder).

In training the encoder runs on the host only for the offline cache pass: `memory
="cpu"` (or `TEXT_DEVICE="cpu"`) with `TEXT_CACHE` online / off is a knob error.

## Encode-ahead

With the online text cache, `TextAhead` wraps the loader: every `ahead` host batches it
encodes the cache misses of all of them in one pass, keeps the results on the host or
the device until the batcher takes them, and reports the position of the first batch
not yet handed out as its checkpoint state (a resume repeats nothing and skips
nothing). Under `stream` / `offload` one pass moves the weights once for `ahead`
batches.

## Reporting

`TextEncodeStats` logs `te_time` (s per step), `te_share` (of the wall time),
`te_tokens_per_s`, `te_h2d_gb`, `te_tflops` and `te_mfu` (TE FLOPs over the same wall
time as the DiT `mfu`, so the two add up). The encoder marks the VRAM timeline before
(`step`) and after (`te`) every pass.

Trainer step (`scripts/bench/step/step.py --text-weights`, RTX 5090, `low` profile,
random-weight DiT at 1 MP, one caption per step, online text; TE ms per step, peak
GiB of the text-encoder pass / of the DiT step):

| TE memory | ahead | TE ms per step | TE pass peak (alloc) | DiT step peak (alloc) | H2D GB per pass |
| --- | --- | --- | --- | --- | --- |
| resident | 1 | 44 (median) | 20.1 | 20.7 | 0 |
| offload | 1 | 533 | 18.2 | 5.9 | 13.8 |
| stream | 1 | 495 | 6.9 | 6.0 | 13.8 |
| offload | 8 | 68 (mean) | 18.2 | 5.9 | 13.8 |
| stream | 8 | 100 (mean) | 6.9 | 5.9 | 13.8 |

One caption per pass is weight-copy bound; `ahead` amortises the copy over the passes'
batches. Sampling (`scripts/sample/sample.py`, 2 x 1024^2, 8 steps): generate 14.9 s /
14.3 s / 13.9 s and peak 14.3 / 22.8 / 29.1 GiB with the encoder streamed / offloaded /
resident.

## Quantization

`precision`: `int8` / `fp8` weight-only (per output channel through `QuantWeight` and
the `quant_linear` kernels, or per `quant_group` input columns, dequantized per call);
`fp8_compute` (`fp8_recipe` row / tensor / block) and `mxfp8`: fp8 compute through
`ComputeWeight` and the tuned `fp8_linear` op (weights quantized once on the compute
device, activations per call; sm89+; [quant-kernels.md](quant-kernels.md#fp8-compute)).
Linears with fewer than 256 input or output features stay dense, as do fp8-compute
linears whose sizes are not multiples of 128.

Qwen3.5-9B on an RTX 5090 (`scripts/probe/fp8_inference.py`, 5 short prompts):
resident weights 14.8 GiB in bf16, 8.38 GiB in fp8_compute, 8.79 GiB in mxfp8;
embeddings against the host fp64 encode, cosine 0.99984 (bf16) / 0.99635 (row) /
0.99604 (tensor) / 0.99654 (block) / 0.99593 (mx), abs mean 0.022 / 0.118 / 0.124 /
0.115 / 0.126. At that size every GEMM is launch-bound and the fp8 encode takes 73-80
ms against 43 ms in bf16.

### Pre-quantized text encoder

`scripts/export/text_quant.py` writes the quantized linears of one precision (`int8` /
`fp8` weight-only per channel, `fp8_compute` with `FP8_RECIPE`, `mxfp8`) to one
safetensors file: `<n>.weight` holds the codes, `<n>.weight_scale` the recipe's scale,
`<n>.bias` the bias. Every other tensor is in `SAVE_DTYPE`, and the metadata key
`kohakumoss_text_quant` records the precision, recipe and layer list
(`kohakumoss/quant/stored_text.py`).

The plan field `stored` names the file: `TEXT_ENCODE = {"precision": "fp8_compute",
"stored": "<file>"}`, in training and inference alike. The encoder is built from the
text-encoder dir's config without initialising or reading the dense weights. The stored
bases are attached as `QuantLinear`, and the remaining tensors load from the file, so
nothing is quantized at load. The file's precision must equal the plan's, and
`quant_group` must be 0. The tokenizer and config still come from the weights root.
