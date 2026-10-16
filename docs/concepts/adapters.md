# Adapters

KohakuMoss trains adapters with the [lycoris-lora](https://github.com/KohakuBlueleaf/LyCORIS)
package: its algorithms, parameter layouts, initialisation, preset semantics, save format
and fused kernels. `kohakumoss.adapters` is a custom applier around it.

## Algorithms

`ALGO` | LyCORIS module | notes
--- | --- | ---
`lora`, `locon` | `LoConModule` | `DORA=True` adds the weight-decompose epilogue
`loha` | `LohaModule` | DoRA available
`lokr` | `LokrModule` | `factor`, `decompose_both`, `full_matrix` via `ALGO_KWARGS`; DoRA available
`oft` | `DiagOFTModule` | `DIM` is the block size bound; `constraint`, `rescaled`
`boft` | `ButterflyOFTModule` | needs a power-of-two factorisation of each output width
`glora` | `GLoRAModule` | ΔW = W·a1·a2 + b1·b2
`ia3` | `IA3Module` | `train_on_input`
`dylora` | `DyLoraModule` | random rank while training, every rank in eval
`tlora` | `TLoraModule` | per-token rank masks from `adapters.tlora.set_token_mask`
`full` | `FullModule` | trains the weight itself; `ADAPTER_MODE="merge"`

`TRAIN_NORMS=True` adds a trained delta to the norms the preset covers
(`adapters.norms.NormDelta`, a parametrization: `weight = original + m·w_norm`).

## Targeting

`PRESET` is resolved by `adapters.presets.resolve_preset` into a full LyCORIS preset
(`target_name` regexes over module paths, which are the checkpoint key paths):

base | layers
--- | ---
`attn` | joint `layers.N.proj.{q,k,v,o}`
`attn-mlp` | + joint `ffn_up_gate`, `ffn_down`
`full-block` | + text-refiner block linears
`all` | every linear of the transformer

Modifiers append: `+text-refiner`, `+modulation` (time embedding, shared and head
modulation), `+embed` (video embedder, final linear). A toml path or a dict with any
LyCORIS preset key (`target_name`, `name_algo_map`, `module_algo_map`, `exclude_name`,
`use_fnmatch`) is used as-is; every key is set explicitly on each call because
`LycorisNetwork` keeps presets as class state.

## Binding

`apply(model, spec)` builds LyCORIS modules on the matched linears (the target layers
are `MossLinear`, an `nn.Linear`), registers each as the child `lyco` of its linear
(FSDP sharding and weight streaming see it with its layer), and installs each layer's
forward once. Per layer the bound forward is one of:

- the LyCORIS fused op of the kind — `bypass` adds the adapter path to the base output,
  `merge` builds ΔW and runs one GEMM;
- the DoRA path for decomposed algorithms: W' = DoRA(W + ΔW), the decompose step being
  LyCORIS's `weight_decompose` op;
- the LyCORIS module's eager body when dropout is configured, on CPU / XPU, or (pinned
  backend only) where the layer is outside a fused kernel's scope (LoHa rank > 128,
  LoKr factors above the kernel bound, OFT block > 32).

How a fused op runs depends on `KERNELS.lycoris`:

- `auto` (default): each op is a `bind.race.TierSlot` bound to LyCORIS's measured,
  precision-gated tier race (`lycoris.kernels.tier`). On the first call of a shape
  bucket (LyCORIS's token bucket of the flattened rows, activation dtype, grad pattern)
  the slot asks `tier.best` with the real operands, which races every fused backend of
  the device, the compile tier, the eager body and the op's exact extras (LoHa's rank
  product) on a miss of LyCORIS's tuning table, rejects any candidate outside the
  eager body's error bar against fp64, and records the fastest. The slot keeps that
  arm; later calls of the bucket are one dict lookup and the arm. A decision LyCORIS
  holds provisionally (an all-zero reference: freshly zero-initialised factors) is
  asked again on every call — LyCORIS's own re-race schedule — until it is recorded.
  Ops outside a fused kernel's scope get LyCORIS's compile or torch tier. IA3 and
  GLoRA have no tier op and run the device's preferred fused backend.
- a backend name (`triton` | `tilelang` | `compile` | `torch`): every op calls that
  backend's kernel (`lycoris.kernels.autograd.*`) directly, with no measurement.

The base output always comes from `MossLinear._impl`, so quantized or streamed base
storage keeps working. Changing the multiplier (`set_multiplier`), the adapter state
(`load_state_dict`) or the base storage re-binds and builds fresh slots; factor dtypes
and `requires_grad` flags are part of a slot's bound state, so changing them calls for
`AdapterSet.bind()`. The device's backend order is LyCORIS's
(`lycoris.kernels.dispatch.resolve_backend`): tilelang first on sm70/sm75, triton from
sm80, eager on CPU and XPU.

## Quantized and streamed bases

When a layer's base lives in storage (`MossLinear.set_base`), its dense parameter is
empty and every adapter path reads the base through `MossLinear.base_weight()`; the
base matmul stays `MossLinear._impl` (the storage's own `linear`). Such layers always
bind the bypass form, since nothing can be merged into the storage; `merge_into`
refuses them, `ALGO="full"` is rejected, and merged exports dequantize through
`base_weight()`. Algorithms whose initialisation reads the weight (DoRA, T-LoRA, OFT,
BOFT) see a CPU dequantized copy during `apply` only.

## Multiplier

The multiplier scales ΔW. For DoRA it moves the decomposed weight linearly toward the
base weight, `W + m·(W' − W)`, which is how ComfyUI applies strength; for OFT/BOFT it
interpolates each rotation toward identity, `m·R + (1 − m)·I`.

## State and export

- `state_dict()` / `save_state()` hold the raw parameters by LyCORIS key (exact resume;
  the file also records the spec).
- `export(path, fmt)` writes `lycoris` (LyCORIS's own state dicts), `comfy` (see
  `docs/guides/export-to-comfy.md`), `merged` (one bare-key transformer file) or
  `diffusers` (shards + index).
- `row_scales` (from `model.numerics.output_row_scales` of a folded model) divides
  ΔW rows so files are always in unfolded space.

## Inference

`load_adapters(model, [(file, multiplier), ...], mode)` reads LyCORIS files,
ComfyUI-style files (`lora_unet_` or `diffusion_model.` keys) and training states:

- `bypass` stacks each adapter's forward on the previous one (low-rank paths stay
  factorised; DoRA, dense diffs, DyLoRA and OFT at multiplier ≠ 1 apply a precomputed ΔW);
- `merge` adds ΔW into the base weights.

`remove()` restores every touched weight and forward bit-exactly. `row_scales` places
unfolded files onto a folded model.
