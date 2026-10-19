# Quantized frozen-base kernels

The GEMMs that serve a frozen base weight stored below 16 bits. Two families:

* **weight-only** (`base_precision` / TE `precision` `int8`, `fp8`): the weight is
  stored quantized and dequantized inside a 16-bit GEMM ([weight-only](#weight-only));
* **fp8 compute** (`fp8_compute`, `mxfp8`): the weight is stored quantized once and the
  GEMM itself runs on e4m3 operands, the activations quantized per call
  ([fp8 compute](#fp8-compute)).

## Choosing a base precision

`BASE_PRECISION` (the DiT, in training and sampling) takes these values;
`TEXT_ENCODE["precision"]` (the text encoder) takes `compute`, `int8`, `fp8`,
`fp8_compute` and `mxfp8` ([text-encoder.md](text-encoder.md#quantization)):

| value | family | arch | frozen-base bytes per weight | trainable base |
| --- | --- | --- | --- | --- |
| `compute` | none (the `PRECISION` dtype) | every | 2 (16-bit) | yes |
| `int8` | weight-only | every (eager on CPU) | 1 + a per-row fp32 scale | no |
| `fp8` | weight-only | sm89+ | 1 + a per-row fp32 scale | no |
| `quant` | weight-only: `fp8` where the arch has fp8 GEMMs, else `int8` | every | 1 + a per-row fp32 scale | no |
| `fp8_compute` | fp8 compute, scale granularity `FP8_RECIPE` | sm89+ | about 1.0; training with `FP8_DGRAD="fp8"` adds the W^T copy ([input gradient](#input-gradient-training)) | yes, through fp8 copies ([full finetune](#full-finetune)) |
| `mxfp8` | fp8 compute, 1 x 32 e8m0 scales | sm89+ (native block-scaled MMA on sm100+) | about 1.03; the same W^T copy in training | yes, through fp8 copies |

On Ada (sm89) `fp8_compute` with the `row` recipe is the fast format; `mxfp8` there
keeps its weight bytes and costs compute time ([measured](#measured-rtx-5090-sm120-bf16-activations)).

## Source

Every quantizer (`quantize_model` for the DiT base, `quantize_text_linears` for the
text encoder, and the two export scripts `fp8_quant.py` / `text_quant.py`) quantizes a
weight from the checkpoint's stored values, not from its live copy
(`kohakumoss.quant.source.source_weight`):

- The stored values are read one tensor at a time (safetensors headers first) and
  multiplied in fp32 by the output-row multiplier of any fold applied since the load
  (`kohakumoss.model.io.quant_source.fold_since_load`: the online fp16 fix, against the
  fold an fp16-fix file already carries).
- They replace the live weight only when they round to it exactly
  (`(stored * fold).to(live.dtype) == live`). Otherwise the live weight is quantized: a
  delta merged into it, a different checkpoint, a model not loaded from a file.
- A weight stored in the live dtype with no fold since the load is not read again (the
  live copy is already its stored values).
- The lookup runs once per weight, when the model is quantized (load / memory plan /
  export), never per step. Each quantizer logs one INFO line per model:
  `base fp8: quantized <n> linears: from the stored values <n>` (the DiT) or
  `text encoder fp8: ...`, with the live-weight counts by reason (stored in its dtype,
  no stored values, not the stored values' cast); the report carries the same counts
  (`from_source`, `sources`).

So the codes and scales depend on the checkpoint, never on the compute dtype: a bf16
checkpoint quantizes the same at fp16 compute (sm75, fp16 training) as at bf16, and a
pre-quantized file holds the codes of quantize-at-load. The fp16 cast changes the bf16
values below fp16's smallest normal (6.1e-5); quantizing that cast instead moves about
1e-5 of the fp8 codes by one step (MOSS-Image `layers.0.proj.q`: 94 of 9.4M codes in
`fp8` / `fp8_compute` row, 45 in block, 227 in `mxfp8`; Qwen3.5 `layers.0.mlp.down_proj`:
1392 of 50M in `fp8`); `int8` codes do not move. Cost: when the stored dtype differs
from the compute dtype (or a fold was applied since the load), one more read of each
quantized weight from the checkpoint and one fp32 copy of it on the quantizing device.

An fp16-fix file stores its weights in its own `SAVE_DTYPE`. Written at `fp16`, its
stored values are the fp16 cast of the bf16 weights (folded), and a quantized base
built from it holds the codes of that cast; written at `bf16` it keeps the bf16 values.

## Weight-only

`kohakumoss_kernels.quant` serves `quant_linear(x, q, scale, bias, fmt)`, the
op `kohakumoss.quant.QuantWeight.linear` resolves once per weight (`bind`).

### Storage

`q` is `[N, K]` (`int8`, or `float8_e4m3fn` for `fmt="fp8"`), `scale` is an
fp32 per-row vector `[N]`, the weight is `q * scale[:, None]`. This is the
layout `QuantWeight.from_weight` produces (int8: `round(w / s)`,
`s = max|w_row| / 127`; fp8: `s = max|w_row| / 448`).

### What runs

| arch | int8 | fp8 |
| --- | --- | --- |
| sm70 / sm75 | TileLang weight-only GEMM (fp16) | - |
| sm80 / sm86 | Triton weight-only GEMM (fp16 / bf16) | - |
| sm89 / sm120 | Triton weight-only GEMM (fp16 / bf16) | Triton |
| XPU | Triton weight-only GEMM (fp16 / bf16) | - |
| CPU | eager reference | eager reference |

* **Forward** `y = (x @ q^T) * scale + bias`: the int8 / e4m3 tile is
  converted to the activation dtype in registers or shared memory (exact for
  both formats), multiplied on the MMA units with an fp32 accumulator, and the
  scale and bias are applied to that accumulator before one rounding at the
  store.
* **Input gradient** `dx = dy @ (q * scale[:, None])`: the scaled tile is
  rounded once to the activation dtype (the same rounding the
  dequantize-then-matmul reference applies) and accumulated in fp32.
* **No weight gradient** (the base is frozen); a trainable bias gets the fp32
  column sum of `dy`.
* **Saved for backward**: only `q` and `scale`. Every registered entry, the eager
  one (CPU, XPU, a device without a built kernel) included, is the same custom op
  (`make_quant_linear`); the eager entry's launchers are the reference dequantize +
  matmul, so its forward and input gradient equal `quant_linear_reference`, while no
  16-bit copy of the weight stays alive from forward to backward.

### Selection

* **Candidates.** Per (device, token bucket, N, K, dtypes, bias) the tuner times every
  kernel tile config **and** the two PyTorch paths (dequantize + `F.linear` eager, and
  the same under `torch.compile`) and keeps the fastest. A kernel therefore only runs
  where it beats both PyTorch paths on that card; elsewhere the op is the PyTorch path.
* **Binding.** The choice is resolved once per key and held in a dict, so the per-call
  cost is one lookup. Only a settled choice (`tuning.decided`) is held.
* **Where nothing is timed.** A call where the tuner may not time (inside a trace, or
  the forward GEMM inside a checkpoint recompute) runs the planner's first candidate for
  that call alone, and a later call where the tuner may time measures the key. Outside
  the tuning windows ([tuning.md](tuning.md#tuning-windows)) a missing key binds its
  first candidate untimed, with one WARNING line.
* **Backward.** The dgrad GEMM runs inside the autograd backward and launches no
  autograd itself, so it is timed there (`in_backward=True`).

The op is a `torch.library` custom op (`kohakumoss::quant_fwd` /
`kohakumoss::quant_dgrad`) with fake implementations and registered autograd,
so it stays in a `torch.compile(fullgraph=True)` graph.

### Checks

* `tests/kernels/quant/test_quant.py`: forward, input and bias gradients against an
  fp64 reference with the shared precision check (stats vs PyTorch, see
  [fp16.md](fp16.md#error-statistics)), off-tile shapes, fullgraph compile, and a
  scale-on-the-wrong-axis variant the check must reject.
* `scripts/bench/kernel/quant.py`: kernel time of every backend against eager
  and compiled dequantize+matmul at the DiT layer shapes, with the path the
  tuner selected for each case.

## Rotation

`kohakumoss_kernels.hadamard` defines the ConvRot rotation of the ComfyUI int8
checkpoints. H_g is the normalised Kronecker power of H4 = [[1,1,1,-1],[1,1,-1,1],
[1,-1,1,1],[-1,1,1,1]], for g = 4^j. It is the regular Hadamard comfy-kitchen builds,
not the Sylvester one. H_g is symmetric and involutory, and its entries are +-2^-j,
exact in fp16 / bf16.

The op `group_hadamard(x, group)` is x . blockdiag(H_g) over the last dim. It is its
own inverse and its own adjoint, so its backward is the op itself.

| backend | arch | what runs |
| --- | --- | --- |
| eager (`group_hadamard_eager`) | every | fp16 / bf16: a 16-bit matmul with H_g, fp32 accumulation, one rounding; any other dtype: j fp32 4-point butterfly stages (`y_k = sum(x) - 2 x_(3-k)`) and the 2^-j scale |
| triton | sm70+ (the sm70 / sm75 path too), XPU | one [BR, C] tile per program in fp32 registers, the same butterflies, one rounding; fp16 / bf16 / fp32 |
| compile | where inductor builds | torch.compile of the eager reference |

There is no TileLang entry. A TileLang version that ran each group as a GEMM against
H_g measured slower than eager on the RTX 2080 Ti and the RTX 3090 (0.27 vs 0.20 ms
at [4096, 3072] on the 2080 Ti), since g = 256 makes it compute-bound. The Triton
butterflies are memory-bound and serve sm70 / sm75 as well.

* **Selection.** Each kernel entry is chosen per shape against the compile path, the
  same way as the block ops ([tuning.md](tuning.md#kernel-vs-compile)).
* **Numerics.** Every path computes x . H_g from the 16-bit (or fp32) input with fp32
  sums and rounds once to the input dtype. The 16-bit inputs times +-2^-j are exact,
  so the fp32 matmul accumulation and the fp32 butterflies differ only in summation
  order. The precision tables bear this out: kernel and eager agree to three digits
  on every mean and median statistic. They differ by up to 1.2x on `abs_max`, and
  only on the relative statistics of near-zero outputs (`rel_max` of the fp32 tables).
* **Cost.** The op reads and writes x once. The Triton kernel rotates a [16384, 3072]
  x in 0.243 ms on the RTX 3090 (about 830 GB/s) and in 0.41 ms on the RTX 2080 Ti
  (about 490 GB/s).

## Rotated int8 linear

`kohakumoss_kernels.quant.rotated` serves `quant_linear_rotated(x, q, scale, bias, fmt,
group)`: the ConvRot weight of a ComfyUI file is stored rotated, W R ~ q * scale with
R = blockdiag(H_g) over K. So:

* forward: `y = quant_linear(x R, q, scale, bias)`;
* input gradient: `dx = (dy (q * scale)) R`;
* bias gradient: the fp32 column sum of dy; the base gets no gradient.

The op is a custom op (`kohakumoss::quant_fwd_rot` / `kohakumoss::quant_dgrad_rot`) whose
autograd saves only `q` and `scale`, so it stays in a `torch.compile(fullgraph=True)`
graph. `kohakumoss.quant.QuantWeight` binds it once per weight (`fmt`, `group` static).

| backend | what runs |
| --- | --- |
| eager | `group_hadamard_eager`, then the dequantize + matmul reference (`quant_linear_rotated_reference`) |
| triton (sm80+, XPU) | per shape the tuner times `separate` (the Triton rotation, then the plain quant kernel, which picks its own tile or PyTorch path) against the fused tiles: the forward rotates each [BM, g] tile of x in fp32 registers in the prologue, and the dgrad rotates its fp32 accumulator tile in the epilogue |
| tilelang (sm70+) | the registry's `group_hadamard` for the device (Triton or compiled, chosen per shape), then the TileLang int8 kernels |

* **Numerics.** The fused forward rounds the rotated tile to the activation dtype
  before the MMA, the same bytes the separate pass writes. The fused dgrad rounds once,
  after the rotation; the separate dgrad rounds before and after it.
* **Cost of the separate pass.** One extra read and write of x per forward (of dx per
  dgrad). The fused forward has no extra pass, but rotates every x tile once per output
  tile, N / BN times. Measured on the RTX 3090 ([quant-rotate.md](../performance/quant-rotate.md)):
  * the separate pass costs 1-7% of the int8 GEMM in the forward, and 1-14% of
    forward + backward (the smallest GEMM, 3072 x 3072 at 4096 tokens, is the 14%);
  * every fused tile is slower than the separate pass at every DiT shape: about 2x in
    the forward (the redundant rotations run on the CUDA cores) and 1.2-1.3x in the
    dgrad.

  The tuner therefore selects `separate` there; the fused tiles stay candidates, timed
  per card.

### Checks

* `tests/kernels/quant/test_hadamard.py`:
  * the matrix against an fp64 `torch.kron` build;
  * the eager reference against fp64 (fp32 butterflies; 16-bit within one rounding)
    and its gradient;
  * every GPU backend and the compile path against eager and fp64 (stats rule), fp16 /
    bf16 / fp32, forward and backward, under fullgraph compile;
  * negatives: Sylvester H, unnormalised H, non-power-of-4 groups.
* `tests/kernels/quant/test_quant_rotated.py`:
  * the eager composition against fp64 (CPU);
  * every backend, and every Triton config on its own (separate and each fused tile),
    against the eager composition and fp64, forward / dx / dbias, under fullgraph too;
  * negatives: no rotation, the rotation on the output side, and the dgrad rotation
    applied to dy instead of dx.
* `scripts/bench/kernel/rotate.py`: the rotation alone per backend, the plain int8
  linear, the rotated linear as selected, and the tuner's separate vs fused timings per
  DiT shape and token count, with the error against fp64.

### ConvRot bases

`kohakumoss.quant.QuantWeight(q, scale, fmt, rotation)` holds a stored ConvRot weight
(a ComfyUI `int8_tensorwise` file with `convrot`): `linear` is bound once to
`quant_linear_rotated(fmt, group=rotation)` (eager when no kernel serves), and
`weight(dtype)` / `materialize()` give W itself: the fp32 dequantized rows times H_g
(the eager butterflies), rounded once. Adapters (DoRA norms, OFT, merged exports), the
dense fallback and the exports therefore all read the unrotated weight.

* **Streamed.** Under weight streaming the ring carries `q` and `scale`;
  `StreamedQuantWeight` dequantizes the slot in fp32, rotates it with the device's
  `group_hadamard` (resolved once at install) and rounds once to the compute dtype.
* **Folds.** A weight fold that scales a stored base's rows (an fp16-fix fold with
  pre-gate scales) is refused (`fold`); the online fix and the uniform gate fold touch
  only dense parameters.

### W4A8 weights

`kohakumoss.quant.W4A8Weight` holds a ComfyUI `asym_w4a8_int8` layer: `q` [N, K/2]
(two 4-bit codes per byte, even column in the low nibble; a 6-bit layout adds a K/4
plane of the top bits), `s_rel` e4m3 [N, K/16], `s_channel` fp32 [N], a 16-entry
`codebook` (or the uniform grid code - 8) and an optional per-group `correction`. The
codes decode exactly onto an int8 grid, `round(level x s_rel)` clamped to +-127, so the
layer is that int8 grid times `s_channel`, stored rotated (ConvRot 256):

| path | bytes per weight | per call |
| --- | --- | --- |
| `to_int8()` (the decode once) | 1 + 4 / K | the rotated int8 linear |
| packed (`linear`) | 0.5 + 1 / 16 + 4 / K | the decode (compiled on CUDA / XPU, eager on the host), then the rotated int8 linear |

Both give the same int8 values, so the same outputs. A text encoder chooses between them
by `TEXT_ENCODE["w4a8"]` ([text-encoder.md](text-encoder.md)); a streamed DiT base is
decoded to int8 before it enters the ring. A layer with a `correction` has no int8 form
and runs its dense weight. Costs: [quant-rotate.md](../performance/quant-rotate.md#w4a8-decode).

### Embedding

`kohakumoss.quant.QuantEmbedding` is an int8 embedding table stored per-row scaled and
rotated over the hidden dim (the ComfyUI text encoders' `embed_tokens`). A lookup
gathers the int8 rows and their scales, dequantizes them in fp32, rotates them back with
the device's `group_hadamard` and rounds once. Under the text encoder's offload / ring
placements the table stays on the host and `lookup` moves only the gathered int8 rows
and scales to the device.

### Stored formats

A checkpoint that stores quantized bases (a KohakuMoss fp8 file, a ComfyUI int8 / fp8 /
W4A8 file) attaches them at load. The memory plan's `base_precision` then decides
(`memory.apply.settle_stored`):

| `base_precision` | stored bases |
| --- | --- |
| `auto` (every profile's default) | kept as stored |
| the stored format (e.g. `int8`) | kept; the other linears quantize to it |
| anything else (`compute` included) | dequantized to dense weights first (one INFO line), then the plan applies |
| any, with a trainable base | dequantized (`compute` / fp8 compute); `auto` is refused, naming the `compute` escape |

`model.io.load.stored_precision(path)` reads the stored format from the headers and
quant records (no weight read): the planner's estimate counts its bytes under `auto`
(`Workload.stored_precision`), and the knob checks refuse a full finetune of a stored
file under `auto`.

## fp8 compute

`kohakumoss_kernels.fp8` serves `fp8_linear(x, q, scale, aux, q_t, scale_t, aux_t,
bias, recipe)`, the op `kohakumoss.quant.compute.ComputeWeight.linear` binds once per
(device, dtype). The design follows KohakUwULLM's MXFP8 path (its 5090 kernels and
measurements are the sm120 reference) and the OCP MX v1.0 specification.

### Knobs

| where | knob | values |
| --- | --- | --- |
| DiT (`BASE_PRECISION` / `FP8_RECIPE` / `FP8_DGRAD` in training and sampling, `MEMORY_OVERRIDES`; `memory_overrides` in Python) | `base_precision` | `fp8_compute`, `mxfp8` (plus the weight-only `int8`, `fp8`) |
| | `fp8_recipe` | `row` (default), `tensor`, `block`; `fp8_compute` only |
| | `fp8_dgrad` | `fp8` (default), `16bit`; training only |
| text encoder (`TextEncoderPlan`, `TEXT_ENCODE` in training and sampling) | `precision` | `fp8_compute`, `mxfp8` |
| | `fp8_recipe` | `row`, `tensor`, `block` |

- Both need CUDA sm89+: the memory-plan and text-plan validation name the arch
  otherwise (sm70 / sm75 / sm86, XPU and the host are rejected).
- With a trainable base (full finetune), `base_precision` fp8 compute trains the 16-bit
  weights through fp8 copies ([full finetune](#full-finetune)), with the weights
  resident (`stream="off"`).

### Formats and recipes

An operand is `[rows, K]` with K the contraction axis; values are e4m3 (`float8_e4m3fn`,
max 448). The scale granularity is the recipe:

| recipe | activation scale | weight scale | scale type |
| --- | --- | --- | --- |
| `tensor` | one per tensor `[1]` | one per tensor `[1]` | fp32 |
| `row` | per token `[M]` | per output channel `[N]` | fp32 |
| `block` | 1 x 128 along K `[M, K/128]` | 128 x 128 blocks `[ceil(N/128), K/128]` | fp32 |
| `mx` (`mxfp8`) | 1 x 32 along K `[M, K/32]` | 1 x 32 along K `[N, K/32]` | e8m0, biased uint8 |

* **fp32 scales** are `max(amax, 1e-12) / 448` and `q = e4m3(clamp(x / s))`, both
  correctly rounded divisions.
  * PyTorch divides a tensor by a Python scalar through the reciprocal (one ulp off for
    some amax); the eager reference divides by a tensor and the Triton kernels use
    `div_rn`, so kernel and reference bytes are equal.
  * Triton's fp32 -> e4m3 conversion on sm89 goes through fp16 (two roundings, a
    different byte for values near a rounding midpoint); the kernels round to the e4m3
    grid in fp32 first (round-to-nearest-even by the magic-number add), which makes that
    conversion exact on every arch.
* **The mx exponent rounds up**: the smallest `e` with `2^e * 448 >= amax`, floored at
  -127, from the bits of amax (`amax = m 2^E`: `e = E - 8`, or `E - 7` when
  `m > 1.75`), so a block's amax never lands above 448 and no value clips. The
  round-to-nearest OCP exponent clips the block maximum, the failure NVIDIA and
  KohakUwULLM measured as divergence. `q = e4m3(x * 2^-e)` (an exact scaling).
* **`aux`** is the vendor copy of an mx weight scale: the 128-row x 4-column tiles,
  32 x 4 x 4 interleaved (`SWIZZLE_32_4_4`), zero-padded to whole tiles.
* **Layer sizes:** fp8 compute layers need N and K multiples of 128; others (and
  KEEP_FP32 modules, and linears under 256 features) stay dense and are counted in the
  quantization report (`skipped`).

The eager reference is `kohakumoss_kernels.fp8.formats.quantize` / `dequantize`; its bytes
are witnessed against an independent numpy formula (e4m3 rounding from the bit
layout, the scales block by block) in `tests/quant/test_fp8_formats.py`.

### Weights and activations

- **Weights.** `ComputeWeight.from_weight` quantizes the weight once, when the memory
  plan is applied (on the compute device, then placed), and stores only `q` (1 byte per
  weight), the scale and `aux`: about 1.0 (tensor / row / block) and 1.03 (mx) bytes per
  weight against 2 for bf16. The memory planner counts those bytes (`PRECISION_BYTES`),
  the W^T copy in training, and the per-call fp8 activation and 16-bit fallback
  buffers. With weight streaming (`stream="ring"`) the ring carries the quantized parts
  (`StreamedComputeWeight`).
- **Activations** are quantized on the fly inside each GEMM path (one bandwidth-bound
  pass over `x`, fixed tiles: the token count changes every step); the 16-bit copy of
  the weight is never built on the fp8 paths.
- **Where the quantizer sits.** In the consumer, as in KohakUwULLM's dense path: a LoRA
  on the same linear needs the 16-bit input anyway, and the path picked per shape
  decides the layout (natural scales, the vendor swizzle, or quantized inside the GEMM
  tile loop). The cost this leaves is the q / k / v projections quantizing one
  normalized input three times: 0.07-0.14 ms per extra pass at 4k-16k tokens on the
  5090, 2-5% of a DiT step.

### Input gradient (training)

A frozen base needs only `dx = dy W`:

| `fp8_dgrad` | bytes per weight | input-gradient GEMM |
| --- | --- | --- |
| `"fp8"` (default) | 2 | the base also stores W^T quantized along N (the contraction axis of that GEMM, as in KohakUwULLM: an MX block must run along the contraction axis, so the forward copy cannot serve the input gradient); `dy` is quantized e4m3 with the same recipe and the GEMM runs on the fp8 paths |
| `"16bit"` | 1 | `dy @ dequant(W)` in 16 bits |

There is no weight gradient (adapter math stays 16-bit in LyCORIS); a trainable bias
gets the fp32 column sum of `dy`. The activation gradient is straight-through (the
quantizer passes `dy @ W`).

### Full finetune

KohakUwULLM trains its full models in MXFP8, and so does a full finetune here:

- `base_precision` fp8_compute / mxfp8 with a trainable base keeps the 16-bit weight as
  the trained parameter and attaches a `ComputeCache` (the forward fp8 copy and, with
  `fp8_dgrad="fp8"`, the W^T copy) to every fp8-compute-shaped linear
  (`MossLinear.set_compute`).
- The forward and the input gradient run in fp8 (`fp8_linear_train`); the weight
  gradient is `dy^T x` on 16-bit operands with fp32 accumulation (x saved in its own
  dtype), as KohakUwULLM keeps WGRAD 16-bit.
- The copies are rebuilt from the weights after every optimizer step
  (`MemoryEngines.after_step`); they are plain attributes outside the module state, so a
  checkpoint holds the 16-bit weights alone.
- Weights stay resident (`stream="off"`). Bytes per weight: the 16-bit weight plus 1
  (`fp8_dgrad="16bit"`) or 2 (`"fp8"`).

### Pre-quantized checkpoints

`scripts/export/fp8_quant.py` writes a transformer whose fp8-compute linears are stored
as `<n>.weight` (e4m3 q, quantized from the checkpoint's stored values: [source](#source))
+ `<n>.weight_scale` (the recipe's scale), every other tensor
16-bit, the recipe and layer list in the safetensors metadata (`kohakumoss_fp8`).

- `load_backbone` attaches those bases directly; no 16-bit copy is built. The file is
  `TRANSFORMER` in the sample script and the single-file `WEIGHTS` in the train script.
- A memory plan's quantization skips layers that already carry a base.
- A frozen-base training run with `fp8_dgrad="fp8"` derives the W^T copy from the
  stored forward copy.
- MOSS-Image base in mxfp8: a 5.33 GB file; loaded onto an RTX 5090 in 1.3 s against
  7.5 s for the bf16 load plus quantization, 5.34 GiB resident, the same forward bit for
  bit.

### What runs

The GEMM form is `c = Q(a) B^T` with B `[N, K]` K-contiguous (the forward: `a = x`,
`B = W`; the fp8 input gradient: `a = dy`, `B = W^T`). GEMM paths per recipe (the
`gemm` field of a tuner config in `kohakumoss_kernels.backends.triton.fp8.linear`):

| recipe | paths (timed per problem) |
| --- | --- |
| `tensor` | `vendor` (`torch._scaled_mm`, tensor-wise scales), `triton` (`scaled_mm`: each BK-deep fp8 MMA result added to a separate fp32 accumulator, the weight scale in the epilogue) |
| `row` | `vendor` (`torch._scaled_mm`, row-wise scales), `triton`, `fused` (one launch: a K prologue takes each row's amax, the main loop quantizes the A tiles) |
| `block` | `promote` (each 128-deep K block's MMA result scaled and added to the fp32 accumulator, DeepGEMM-style), `fused` |
| `mx` | sm100+: `vendor` (`F.scaled_mm` BlockWise1x32 with swizzled e8m0 scales), `dot_scaled` (`tl.dot_scaled`, the native block-scaled MMA), `fused` (A quantized per 32 columns in the tile loop, `tl.dot_scaled`); every arch: `promote` (32-deep blocks; below sm100 on the fp16 MMA of the exactly converted operands, `acc="upcast"`) |
| every recipe | `w16` (dequantize the weight, 16-bit matmul; eager and `torch.compile`) |

- **`fused`** paths quantize their A rows in every program (an N-fold redundancy) and
  win where M is small: one launch instead of two and no activation round trip.
- **The planner's first pick** (what `TUNE="off"` without a stored pick, or a trace,
  runs; `backends/triton/fp8/linear.py` `candidates`):

  | recipe | first pick |
  | --- | --- |
  | `tensor` | `vendor` |
  | `row` | `vendor` on sm90+; `triton` on sm89, where the vendor row-wise GEMM measures 2-3x slower than the Triton one (L4, below) |
  | `block` | `promote` |
  | `mx` | `vendor` on sm100+ when K is a multiple of 128, else `dot_scaled`; `promote` below sm100 |
- **mx `vendor` and `fused`** read whole 128-column tiles (the swizzled activation
  scales, the fused K loop), so the planner offers them only when the contraction length
  is a multiple of 128; an mx problem with K a multiple of 32 only runs `dot_scaled`,
  `promote` or `w16`.
- **`block`** runs on the Triton path: the vendor block (1x128 / 128x128) GEMM exists
  in PyTorch only on sm90.
- **sm89** has no block-scaled MMA: `mx` runs the `promote` path there, on the fp16 MMA
  (see [Precision](#precision)).
- **Activation quantizers:** per row / per tensor / per 128 / per 32 along K in the
  natural layout, and per 32 with the scales written straight into the vendor swizzle
  (the vendor mx path).

### Selection

- Per (device, token bucket, N, K, dtype, recipe, bias, dgrad copy) the tuner times
  every path (each tile config of the Triton paths) on the real call and keeps the
  fastest; a path that cannot build or run there (an unsupported vendor GEMM, a tile
  over the shared memory) is skipped.
- A pick is bound only once settled (`tuning.decided`), exactly as for the weight-only
  op, and the input-gradient GEMM is timed inside the backward.
- The `w16` path is in the timed shortlist, so an fp8 GEMM runs only where it beats the
  16-bit path on that card. Where `w16` is chosen the layer computes `x @ dequant(W)^T`
  (the weight still stored quantized, the activation unquantized: the same function at a
  higher precision).
- `TUNE="off"` takes the stored pick of the card's table, else the planner's first pick
  (the table above).

Before a pick is bound it runs once under no_grad on the call's own inputs:

- a config that cannot run there (a build or resource error, an unsupported-feature
  RuntimeError such as row-wise `_scaled_mm` on some sm89 builds) is dropped for that
  (device, dtype, direction, recipe), logged once, and the next pick is taken;
- a failing Triton tile drops that tile only;
- when no path is left the op raises `BackendUnavailable` and the registry rebinds
  `fp8_linear` to the eager reference (`tests/kernels/quant/test_fp8_probe.py`).

The op is the `torch.library` custom op `kohakumoss::fp8_linear` (with
`kohakumoss::fp8_dgrad` for the input gradient), fake implementations and registered
autograd, so it stays in a `torch.compile(fullgraph=True)` graph.

### Measured (RTX 5090, sm120, bf16 activations)

One linear at the DiT shapes, ms per call including the activation quantizer (best tile
per path; `scripts/probe/fp8_inference.py` and the tuning tables):

| M x N x K | bf16 `F.linear` | row vendor | row triton | mx vendor | mx dot_scaled | block promote | tensor vendor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4200 x 3072 x 3072 | 0.347 | 0.293 | 0.328 | 0.243 | 0.281 | 0.535 | 0.342 |
| 4200 x 16384 x 3072 | 1.718 | 0.961 | 1.033 | 0.760 | 0.930 | 2.027 | 0.970 |
| 4200 x 3072 x 8192 | 0.867 | 0.579 | 0.683 | 0.430 | 0.599 | 1.190 | 0.558 |
| 16600 x 3072 x 3072 | 1.334 | 0.789 | 0.845 | 0.592 | 0.737 | 1.543 | 0.784 |
| 16600 x 16384 x 3072 | 7.077 | 3.457 | 3.659 | 2.478 | 3.293 | 7.582 | 2.495 |
| 16600 x 3072 x 8192 | 3.521 | 2.094 | 2.246 | 1.478 | 2.062 | 3.928 | 1.821 |

- The vendor MX GEMM reaches 2.4-2.9x bf16 at the large shapes.
- The Triton block promote path stays near bf16 (its per-block accumulator scaling, at
  128 K per step), so `block` gains only where `fused` wins.
- The activation quantizer alone costs 0.07-0.14 ms at these shapes (0.30-0.43 ms at
  K = 8192, M = 16600); the q / k / v projections each quantize the same normalized
  input.

NVIDIA L4 (sm89, 72 W; `scripts/bench/kernel/fp8.py`), bf16, the selected path's
forward / input-gradient speedup over 16-bit:

| case | M | row | tensor | block |
| --- | --- | --- | --- | --- |
| 4096 x 3072 x 3072 | 4096 | 2.13 / 1.95 (triton) | 1.91 / 1.65 | 1.79 / 1.63 |
| 4096 x 16384 x 3072 | 4096 | 1.76 / 1.42 | 1.72 / 1.23 | 1.45 / 1.38 |
| 16384 x 3072 x 8192 | 16384 | 1.65 / 1.95 | 1.25 / 1.78 | 1.40 / 1.61 |
| 256 x 3072 x 3072 | 256 | 0.67 / 0.68 | 0.48 / 0.50 | 0.65 / 0.65 |

sm89 has no block-scaled MMA, and its fp8 MMA accumulates too coarsely to match the
error statistics of the MXFP8 reference, so `mx` runs the promote path on the fp16 MMA
there, slower than bf16 `F.linear`. RTX 4090, bf16, forward ms at M = 4096 (best
promote tile):

| N x K | bf16 | promote | `w16` |
| --- | --- | --- | --- |
| 3072 x 3072 | 0.49 | 0.57 | 0.61 |
| 16384 x 3072 | 2.58 | 2.93 | 3.55 |

On Ada `mxfp8` keeps its weight bytes and costs compute time; `row` (Triton) is the
format for Ada.

End to end on the RTX 5090:

| workload | bf16 | fp8_compute row | tensor | block | mxfp8 |
| --- | --- | --- | --- | --- | --- |
| DiT, 1024^2, 20 steps, 2 prompts, CFG packed (s per step) | 0.875 | 0.567 | 0.570 | 0.638 | 0.483 |
| DiT weights (GiB) | 9.66 | 5.01 | | | 5.33 |
| text encoder (Qwen3.5-9B), 32 prompts of about 150 words per pass (ms) | 445 | 265 | | | 222 |

At 5 short prompts every text-encoder call is launch-bound and fp8 is slower than bf16
(the encode times of `scripts/probe/fp8_inference.py`:
[text-encoder.md](text-encoder.md#quantization)).

LoRA training (r16, attn-mlp, 1 MP, MAX_TOKENS 8192, block checkpointing, the text
encoder streamed and encoding online in the same precision, 30 steps, medians from step
8; `gated_residual` pinned to eager in every run):

| base / TE | s / step | samples / s | TE s | DiT s | peak GiB (alloc) | resident GiB (alloc) |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 1.478 | 0.68 | 0.542 | 0.938 | 13.60 | 10.23 |
| fp8_compute row | 1.087 | 0.95 | 0.291 | 0.826 | 13.62 | 10.24 |
| mxfp8 | 1.074 | 0.94 | 0.313 | 0.761 | 14.26 | 10.89 |
| mxfp8, `fp8_dgrad="16bit"` | 1.154 | 0.88 | 0.311 | 0.843 | 9.28 | 5.90 |

- The losses track bf16 (step 30: 0.6125 / 0.6196 / 0.6212 / 0.6206).
- With the fp8 input gradient the base holds 2 bytes per weight, as bf16 does; `16bit`
  keeps 1 byte and saves 4.3 GiB.
- At 512^2 / 2048 tokens per step every fp8 call is launch-bound and the DiT part is
  slower than bf16 (0.37 s -> 0.44-0.46 s) while the text encoder still halves.

### Precision

**References.** The fp64 reference of an fp8-compute layer is the fp64 product with its
quantized weight (activations unquantized); PyTorch at the same precision is quantize,
dequantize, `F.linear` (`fwd_reference` / `dgrad_reference`; `F.linear(x, dequant(W))`
for the `w16` path).

**Inside the kernels** the MMA takes e4m3 (or exactly converted fp16) operands,
accumulates in fp32, applies scales and bias to the fp32 accumulator and rounds once at
the store to the output tensor's dtype; the `vendor` and `w16` paths are plain PyTorch
calls.

**Accumulation.** The Triton and fused row / tensor GEMMs add every BK-deep (64-256) fp8
MMA result into a separate fp32 accumulator (`acc="promote"`) instead of chaining the
MMA's own accumulator across K:

- on sm89 the chained fp8 MMA accumulator loses precision with K (3-6x the error of
  PyTorch at the same operands at K = 3072-8192);
- on sm120 the two are identical and cost the same;
- the block and mx promote paths scale each K block into fp32 by construction.

**Ada's fp8 MMA** is coarser than fp32 even into a zero accumulator: one 32-deep block
of e4m3 products comes out with a mean relative error of 1.7-2.0e-4 against the exact
product (the fp16 MMA of the same operands: about 1e-6, fp32 rounding; on sm120 the fp8
MMA matches it), unbiased and uniform over rows, columns and K blocks.

- **mx on sm89.** The mx scales are powers of two, so PyTorch's dequantize + `F.linear`
  multiplies the exact operands with fp32 accumulation and rounds only the output. On
  the fp8 MMA the mx promote path leaves an absolute error floor of about 1e-4 under
  that rounding, which shows on outputs near zero. Below sm100 the mx promote path
  therefore runs the fp16 MMA on the exactly converted operands (`acc="upcast"`;
  e4m3 -> fp16 is exact).

  | M x N x K = 2048 x 512 x 8192, fp16, vs fp64 | abs_mean | rel_mean | rel_max |
  | --- | --- | --- | --- |
  | mx promote on the fp8 MMA | 2.72e-4 | 5.83e-4 | 27.9 |
  | PyTorch | 2.54e-4 | 1.77e-4 | 0.091 |
  | mx promote on the fp16 MMA (`acc="upcast"`) | 2.54e-4 | 1.76e-4 | 0.028 |

- **tensor / row / block on sm89** keep the fp8 MMA: their fp32 scales make PyTorch's
  dequantized 16-bit operands round, which costs more than the fp8 MMA does (row, fp16:
  Triton abs_mean 2.93e-4 against PyTorch 4.96e-4).

**The check.** `tests/kernels/quant/test_fp8_accumulation.py` holds every timed path, at
K = 8192, to no worse mean abs / rel error than plain PyTorch at the same quantized
operands (dequantize + F.linear, and PyTorch's fp8 GEMM `torch._scaled_mm` /
`F.scaled_mm` with the activation dtype as output where the device runs it). An
accumulator rounded every 128 K fails it, and so do, on sm89, the chained Triton path
and the mx promote path on the fp8 MMA; the planner offers sm89 only fp16-MMA mx promote
tiles. Every path is checked in bf16 and fp16, forward and input gradient, at off-tile
shapes (`tests/kernels/quant/test_fp8.py`).

### Checks

* `tests/quant/test_fp8_formats.py`: eager-reference bytes and scales against the
  independent numpy formula per recipe and role, scale layouts, dequantize against the
  per-element product, the vendor swizzle against its index formula; a round-to-nearest
  mx exponent, row scales on the wrong axis and a row-major swizzle must fail.
* `tests/kernels/quant/test_fp8.py`: Triton quantizers bit-equal to the eager reference
  (strided input included), the swizzled quantizer, every path forward + input gradient,
  the registered op (fp8 and 16-bit dgrad, bias gradient), fullgraph compile, the arch
  check, the mx planner leaving out whole-tile paths at a partial K tile; weight scales
  on the wrong rows must fail the error statistics.
* `tests/kernels/quant/test_fp8_probe.py`: a path that cannot run is probed, dropped and
  logged once; no path left falls back to the eager backend.
* `tests/quant/test_fp8_compute.py`: the text-encoder and memory-plan validation per
  arch, and the sample knobs' mapping to the memory plan.
* `scripts/probe/fp8_inference.py`: real weights: text-encoder embeddings and DiT
  velocity per precision against the host fp64 forward of the bf16 weights next to
  bf16, encode / forward / generate time, weight and peak VRAM, decoded images.
