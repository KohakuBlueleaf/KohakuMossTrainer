# Text encoder: kernels, batching, placement

MOSS-Image conditions on the last hidden states of Qwen3.5-9B (32 decoder layers: 24
Gated DeltaNet layers, 8 causal GQA full-attention layers, hidden 4096). The encoder is
frozen; everything here is forward-only under `torch.no_grad`.

This page covers the knobs that place and batch the encoder, the forward and its
layouts, encode-ahead, quantization, reporting, and then the text ops and their kernels.

## TextEncoderPlan

One knob set (`kohakumoss.types.plans.TextEncoderPlan`) serves:

- training: `TEXT_ENCODE`, plus `TEXT_DEVICE` for the device;
- the pipeline: `text_plan=`, and `TEXT_ENCODE` in `scripts/sample/sample.py`;
- the offline cache factory: `kohakumoss.encoders.text.text_encoder`.

| field | values (default first) | effect |
| --- | --- | --- |
| `memory` | `resident`, `offload`, `ring` (`stream` is accepted as the same value), `cpu`, `auto` | placement ([offload.md](offload.md#text-encoder)); the dataclass default is `resident` (`kohakumoss.types.plans`), and training and sampling resolve an unset `memory` to `auto` |
| `free_after_use` | `False`, `True` | drop the model after its first pass |
| `precision` | `auto`, `compute`, `int8`, `fp8`, `fp8_compute`, `mxfp8`, `w4a8` | weight storage and GEMMs ([Quantization](#quantization)); `auto` is what the weights store ([ComfyUI files](#comfyui-files)), `w4a8` a ComfyUI W4A8 file only |
| `w4a8` | `auto`, `int8`, `packed` | a W4A8 file's decode ([ComfyUI files](#comfyui-files)) |
| `quant_group` | `0` (per output channel), or any multiple of 16 (e.g. `64`, `128`) | weight-only scale groups of input columns; `int8` / `fp8` only |
| `fp8_recipe` | `row`, `tensor`, `block` | `fp8_compute` scaling |
| `stored` | `""`, a file | a pre-quantized text encoder ([below](#pre-quantized-text-encoder)) |
| `dtype` | `auto` (bf16, or fp16 where the device has no bf16), `bf16`, `fp16` | compute dtype |
| `stream_resident` | `2` | ring slots per layer kind under `ring` |
| `pinned` | `True`, `False` | pinned host copies for offload / ring |
| `host` | `auto`, `copy`, `file` | the host form of what offload / ring keep on the host: copies in the compute dtype, or the checkpoint's file pages; `auto` copies when they fit the host's room ([offload.md](offload.md#host-room)) |
| `batch` | `32` | prompts per kernel call |
| `max_tokens` | `16384` | tokens per kernel call |
| `layout` | `packed`, `padded`, `single` | [Forward and layouts](#forward-and-layouts) |
| `forward` | `kernels`, `hf` | the kernel forward or the transformers modules' own (`single` / `padded` only) |
| `ahead` | `1` | host batches encoded per pass by the online text cache ([Encode-ahead](#encode-ahead)) |
| `ahead_device` | `host`, `device` | where encoded results wait |
| `kernels` | `{}` | a text op -> a backend, e.g. `{"gated_delta_rule_varlen": "triton"}` |

In training the encoder runs on the host only for the offline cache pass: `memory
="cpu"` (or `TEXT_DEVICE="cpu"`) with `TEXT_CACHE` online / off is a knob error.

## Streamed load

`QwenTextEncoder.from_pretrained` (`kohakumoss.encoders.text.load`) never holds a dense
copy of the model in host memory:

1. The text model is built from its config with its parameters on the meta device
   (`accelerate.init_empty_weights`; the rotary buffers are built on the host). The
   checkpoint's tensors are read by the text model's names as views of the
   memory-mapped safetensors (`encoders/text/files.py`: a model dir's shards, a
   ComfyUI file, a `stored` file); nothing is copied until a tensor moves.
2. Stored-quantized layers (ComfyUI, `stored`) attach their bases over the file pages.
3. The W4A8 decode and `memory="auto"` are settled before any weight is read, on the
   meta model's sizes plus the bytes the load adds: the int8 decode, and quantization
   at load, measured by quantizing one meta-device linear of each shape.
4. Module by module (each decoder layer, then the rest) every tensor goes to its home:
   the compute device under `resident` (and a ring's final norm), else it stays the
   file page, its compute dtype recorded when the file stores another (an fp16 encoder
   from the bf16 checkpoint) and applied by the copy that moves it. A module with
   linears to quantize goes through the work device in the compute dtype, is quantized
   there (each dense linear freed as it is replaced) and moves to its home quantized; a
   dequantized ComfyUI layer is dequantized on the host one layer at a time.
5. The host form (`host`) is settled from the bytes a copy would add; under `cpu` a
   tensor already in the compute dtype runs from its file page.

The embeddings equal those of a load that reads the whole model to the host first:
every value is the same cast of the same stored tensor.

Peak private host memory (anon + shmem of the process) of load + encode + release,
Qwen3.5-9B fp16 from the bf16 checkpoint, RTX 2080 Ti:

| plan | whole model to the host first | streamed |
| --- | --- | --- |
| `resident` | 16.4 GiB | 2.7 GiB |
| `ring` | 19.2 GiB | 1.1 GiB (`host` file) |
| `offload` | 19.2 GiB | 1.1 GiB (`host` file) |

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
linears, norms' weights and rotary tables, with the [text ops](#ops):

- each decoder layer's block function is bound once from its layer type;
- it runs layer-major over all packs of a pass;
- right padding is exact for real tokens because every layer is causal (causal
  attention, causal conv, forward recurrence).

### Encode throughput

`scripts/bench/textenc/textenc.py`, RTX 5090, bf16, 64 captions of 20-160 words = 13.2k
tokens per pass; tokens/s / peak device GiB during the pass. `offload` / `ring` move
13.8 GB of weights per pass and hold nothing between passes:

| forward | memory | layout | batch 1 | batch 8 | batch 32 |
| --- | --- | --- | --- | --- | --- |
| kernels | resident | packed | 4671 / 15.1 | 12540 / 15.3 | 14332 / 16.1 |
| kernels | resident | padded | 4713 / 15.0 | 12137 / 15.3 | 10165 / 16.5 |
| kernels | offload | packed | 3890 / 13.1 | 8797 / 13.4 | 9183 / 14.2 |
| kernels | ring | packed | 4652 / 1.8 | 12778 / 2.1 | 12714 / 2.9 |
| kernels | ring | padded | 4554 / 1.8 | 11761 / 2.1 | 9652 / 3.3 |
| transformers | resident | padded | 2383 / 15.0 | 9325 / 15.6 | 6709 / 17.9 |

`ring` runs at the resident speed from batch 8 on: the layer ring hides the weight
copies behind the compute of a batched pass.

### Accuracy against transformers

The check is the error-statistics check (`tests/_util/error_stats.py`) with
transformers' encode at the same precision as the baseline, in the same batching (its
unbatched encode for one prompt per call, its padded batch for the batched layouts):
the stats table against the fp64 encode is recorded, and a layout fails only on a
regression. The fp64 encode is the kernel forward on the host, whose ops are checked
against independent references.

On real weights (32 captions, 1-300 tokens; `tests/model/test_text_encoder_real.py`,
RTX 5090), abs error / rel error:

| dtype | transformers single | transformers padded | kernels single | kernels packed | kernels padded | packed, one merged segment |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 22.7 / 3.1e5 | 21.1 / 1.2e6 | 10.5 / 6.8e5 | 16.8 / 6.2e5 | 22.5 / 1.0e6 | 76.6 / 4.1e6 |
| fp16 | 1.78 / 1.5e4 | 3.54 / 6.9e4 | 2.07 / 1.1e5 | 7.89 / 1.5e4 | 2.13 / 3.9e4 | |

- The rel error is set by near-zero reference elements, the abs error by the few
  prompts whose rounding differences grow through the 32 layers.
- Either moves by a factor of about 2 with any change of GEMM shape or reduction order
  (transformers' own two batchings differ by 2x in fp16 abs and 4x in rel).
- The packed pass with one merged segment (prompts attending and recurring across their
  boundaries) exceeds both bounds (3.4x abs, 13x rel of transformers' unbatched encode).

## Encode-ahead

With the online text cache, `TextAhead` wraps the loader:

- every `ahead` host batches it encodes the cache misses of all of them in one pass;
- it keeps the results on the host or the device (`ahead_device`) until the batcher
  takes them;
- its checkpoint state is the position of the first batch not yet handed out, so a
  resume repeats nothing and skips nothing.

Under `ring` / `offload` one pass moves the weights once for `ahead` batches.

## Quantization

| `precision` | storage | GEMM |
| --- | --- | --- |
| `int8` / `fp8` | weight-only, per output channel through `QuantWeight` and the `quant_linear` kernels, or per `quant_group` input columns, dequantized per call | 16-bit |
| `fp8_compute` (`fp8_recipe` row / tensor / block), `mxfp8` | `ComputeWeight`, weights quantized once on the compute device, activations per call | the tuned `fp8_linear` op, sm89+ ([quant-kernels.md](quant-kernels.md#fp8-compute)) |

Linears with fewer than 256 input or output features stay dense, as do fp8-compute
linears whose sizes are not multiples of 128.

Each linear is quantized from the text-encoder checkpoint's stored values (its
safetensors tensor, by its weight name), not from the plan-dtype copy the layer loaded
to the work device ([quant-kernels.md](quant-kernels.md#source)): the codes are those of
the bf16 checkpoint whatever the dtype, so an fp16 encoder (sm75) quantizes like a bf16
one. The quantization report counts these layers in `from_source`.

Qwen3.5-9B on an RTX 5090 (`scripts/probe/fp8_inference.py`, 5 short prompts),
embeddings against the host fp64 encode:

| precision | resident weights GiB | cosine | abs mean |
| --- | --- | --- | --- |
| bf16 | 14.8 | 0.99984 | 0.022 |
| fp8_compute row | 8.38 | 0.99635 | 0.118 |
| fp8_compute tensor | 8.38 | 0.99604 | 0.124 |
| fp8_compute block | 8.38 | 0.99654 | 0.115 |
| mxfp8 | 8.79 | 0.99593 | 0.126 |

At that size every GEMM is launch-bound and the fp8 encode takes 73-80 ms against 43 ms
in bf16.

### Pre-quantized text encoder

`scripts/export/text_quant.py` writes the quantized linears of one precision (`int8` /
`fp8` weight-only per channel, `fp8_compute` with `FP8_RECIPE`, `mxfp8`) to one
safetensors file:

- `<n>.weight` holds the codes (quantized from the checkpoint's stored values, so they
  equal quantize-at-load at any `SAVE_DTYPE` and any compute dtype),
  `<n>.weight_scale` the recipe's scale, `<n>.bias` the bias;
- every other tensor is in `SAVE_DTYPE` (`bf16` keeps the checkpoint's values);
- the metadata key `kohakumoss_text_quant` records the precision, recipe and layer list
  (`kohakumoss/quant/stored_text.py`).

The plan field `stored` names the file, in training and inference alike:

```python
TEXT_ENCODE = {"precision": "fp8_compute", "stored": "<file>"}
```

- The encoder is built from the text-encoder dir's config without initialising or
  reading the dense weights. The stored bases are attached as `QuantLinear` over the
  file pages, and the remaining tensors stream from the file, so nothing is quantized
  at load.
- The file's precision must equal the plan's (or the plan's be `auto`), and
  `quant_group` must be 0.
- The tokenizer and config still come from the weights root.

### ComfyUI files

A ComfyUI text-encoder file (`text_encoders/qwen3.5_9b_*`, [weights.md](../guides/weights.md#text-encoder))
streams through `comfy_checkpoint` ([Streamed load](#streamed-load)): the text model is
built from the packaged config without initialising its weights, `model.language_model.*`
streams onto it, the tokenizer is the packaged one.

- `precision="auto"` resolves to the file's format, read from its headers and quant
  records (`text_file_format`): `compute` for the bf16 file, `int8` (int8 ConvRot
  linears) or `w4a8`. Its quantized linears attach as `QuantLinear` over the stored
  base, its int8 embedding as `QuantEmbedding`; nothing is quantized at load. On the
  host (memory `cpu`) `auto` is `compute`.
- Any other precision dequantizes the stored layers into the dense model (one INFO
  line) and quantizes from there. Away from `resident` the dequantized layers are host
  memory of their own (a WARNING when they do not fit the host's room).
- `w4a8` on a W4A8 file: `int8` decodes every layer once at load (on the compute device,
  the result at the layer's home), `packed` keeps the 4-bit codes and decodes per call
  (the compiled decode on CUDA / XPU), `auto` (`default_w4a8`) takes `int8` when the
  decoded encoder fits resident next to the step's reserve on the device and `packed`
  otherwise; then `memory="auto"` chooses the placement on the chosen form's sizes.
- `encoder_spec` (the text cache key) appends the stored layout, e.g.
  `bf16/int8/g0/src:int8-convrot256`; the offline cache factory computes the same spec
  from the headers.
- The kernel prewarm's weightless twin mirrors the loaded quantized modules (their
  shapes, formats and rotations), so it records the rotated int8 calls a live encode
  makes.

## Reporting

`TextEncodeStats` logs:

| metric | meaning |
| --- | --- |
| `te_time` | s per step |
| `te_share` | share of the wall time |
| `te_tokens_per_s` | encoded tokens per second |
| `te_h2d_gb` | host-to-device GB |
| `te_tflops`, `te_mfu` | TE FLOPs over the same wall time as the DiT `mfu`, so the two add up |

The encoder marks the VRAM timeline before (`step`) and after (`te`) every pass.

Trainer step (`scripts/bench/step/step.py --text-weights`, RTX 5090, `low` profile,
random-weight DiT at 1 MP, one caption per step, online text; TE ms per step, peak
GiB of the text-encoder pass / of the DiT step):

| TE memory | ahead | TE ms per step | TE pass peak (alloc) | DiT step peak (alloc) | H2D GB per pass |
| --- | --- | --- | --- | --- | --- |
| resident | 1 | 44 (median) | 20.1 | 20.7 | 0 |
| offload | 1 | 533 | 18.2 | 5.9 | 13.8 |
| ring | 1 | 495 | 6.9 | 6.0 | 13.8 |
| offload | 8 | 68 (mean) | 18.2 | 5.9 | 13.8 |
| ring | 8 | 100 (mean) | 6.9 | 5.9 | 13.8 |

One caption per pass is weight-copy bound; `ahead` amortises the copy over the passes'
batches.

Sampling (`scripts/sample/sample.py`, 2 x 1024^2, 8 steps):

| encoder | generate s | peak GiB |
| --- | --- | --- |
| `ring` | 14.9 | 14.3 |
| `offload` | 14.3 | 22.8 |
| `resident` | 13.9 | 29.1 |

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
- `chunk`: chunks of 64 tokens.
  - `prep_kernel` builds the UT transform `T = (I + A)^-1` (A the strictly lower
    `beta_i k_i.k_j e^(G_i - G_j)`) by 16x16 diagonal inverses and block forward
    substitution.
  - `state_kernel` walks each segment's chunks with the state, writing every chunk's
    start state H and `V' = T (beta v - beta e^G k_n H)`; the first chunk runs before
    the loop (no state product, no H) and each loop step folds the previous chunk into
    the state first, so the last chunk's end state is never formed.
  - `out_kernel` (parallel over chunks) writes
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

- Every matmul takes 16-bit (input-dtype) operands with fp32 accumulation.
- The state and every scale and decay stay fp32; T, the chunk start states H, V' and the
  decayed QK are only ever MMA operands, so they are stored in the input dtype (that
  store is the operand rounding).
- The eager reference is plain PyTorch: the same algorithm with its matmuls on
  input-dtype operands, each output in the dtype PyTorch gives (the input dtype). The
  error statistics (`kohakumoss_kernels.error_stats`: cosine and abs / rel error vs fp64
  next to the eager reference's in the same dtype; a result fails only on a regression)
  compare the kernels against that PyTorch result.
- `recurrent` is vector math only (no MMA) and is checked the same way.
- fp32 inputs time only `recurrent` (MMAs take 16-bit operands only).

Two choices keep the chunked kernels at the eager reference's error:

- Decays are sums of same-sign terms: within a 16-row block a masked cumulative sum,
  across blocks suffix + block totals + prefix. `G_i - G_j` of two long prefix sums loses the small difference in
  the rounding of the large sums.
- softplus is `log1p(exp(x))` (libdevice); `log(1 + exp(x))` loses most digits of
  small softplus values, and the error accumulates along the decay.
