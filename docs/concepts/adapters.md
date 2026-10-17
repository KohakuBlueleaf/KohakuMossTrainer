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
`glora` | `GLoRAModule` | dW = W*a1*a2 + b1*b2
`ia3` | `IA3Module` | `train_on_input`
`dylora` | `DyLoraModule` | a rank prefix drawn per step while training, every rank in eval
`tlora` | `TLoraModule` | per-token rank masks set each step; `ADAPTER_MODE="bypass"` only
`full` | `FullModule` | trains the weight itself; `ADAPTER_MODE="merge"`

`ALGO_KWARGS` keys are checked against the LyCORIS module's constructor (and
`LycorisNetwork`'s): an unknown key, or one with its own knob (`alpha`, `lora_dim`,
the dropouts), is an error. T-LoRA's mask takes `mask_min_rank`, `mask_alpha` and
`mask_max_timestep` (LyCORIS's `compute_timestep_mask` with t = sigma * max_timestep);
these never reach the module. DoRA rejects input dropout (LyCORIS ignores it).

`TRAIN_NORMS=True` adds a trained delta to the norms the preset covers
(`adapters.norms.NormDelta`, a parametrization: `weight = original + m * w_norm`,
summed in fp32 and rounded once to the norm's dtype).

Each algorithm is one `adapters.kinds.Kind` record (dW builders, fused and eager
forward factories, ComfyUI conversion, row-scale tensors, inference stacking class,
LoRA+ side), registered by kind name, LyCORIS module class and `ALGO` name;
`adapters.register(kind)` adds one. See `docs/internals/adapter-kinds.md`.

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
`use_fnmatch`) is used as-is; every key is set explicitly on each call, since
`LycorisNetwork` keeps presets as class state.

On a model whose bases live in quantized or streamed storage, a layer's base is
materialised on the host only when LyCORIS's constructor reads its weight (DoRA,
T-LoRA, OFT, BOFT, `full` initialisation), one layer at a time.

## Binding

`apply(model, spec)` builds LyCORIS modules on the matched linears (the target layers
are `MossLinear`, an `nn.Linear`), registers each as the child `lyco` of its linear
(FSDP sharding and weight streaming see it with its layer), and installs each layer's
forward once. Per layer the bound forward is one of:

- the LyCORIS fused op of the kind: `bypass` adds the adapter path to the base output,
  `merge` builds dW and runs one GEMM on W + dW, summed in fp32 and rounded once;
- the DoRA path for decomposed algorithms: W' = DoRA(W + dW), the decompose step being
  LyCORIS's `weight_decompose` op;
- the LyCORIS module's eager body: while training with input or rank dropout (the
  fused ops carry neither), while training DoRA with `DORA_DETACH`, for GLoRA bypass,
  OFT at multiplier != 1, DyLoRA, T-LoRA, on CPU / XPU, and (pinned backend only) where
  the layer is outside a fused kernel's scope (LoHa rank > `lycoris.kernels.MAX_HADA_RANK`,
  LoKr factors above the kernel bound, OFT block > 32).

Forwards that differ between train() and eval() (dropout, DoRA detach, DyLoRA, module
dropout) re-bind on the mode change, so sampling and validation take the fused path.
The eager bodies run their sub-ops on the bound tier (`lyco.kernel_backend`): the
pinned backend, or the device's preferred one.

`apply` opts LyCORIS into its measured, precision-gated tier mode
(`lycoris.kernels.set_tier_mode("measured")`), sets the compile tier's shape mode
(`LYCORIS_COMPILE_DYNAMIC`) and turns the race off under `TUNE="off"`
(`lycoris.kernels.set_tuning`). How a fused op runs depends on `KERNELS.lycoris`:

- `auto` (default): each op is a `bind.race.TierSlot`. On the first call of a shape
  bucket (LyCORIS's token bucket of the flattened rows, activation dtype, grad pattern)
  the slot asks `tier.best` with the real operands, which races every fused backend of
  the device, the compile tier, the eager body and the op's exact extras (LoHa's rank
  product) on a miss of LyCORIS's tuning table, rejects any candidate outside the eager
  body's error bar against fp64, and records the fastest. The slot keeps the arm once
  `tier.is_settled` holds; later calls of the bucket are one dict lookup and the arm.
  LyCORIS fixes a key's tier at its first call, and a call inside a checkpointed
  forward or a backward takes the first capable tier without a race, so a checkpoint
  recompute reruns what its forward ran. A race on all-zero references (freshly
  zero-initialised factors) is not recorded and its pick stays fixed for the process.
  `AdapterSet.prewarm(rows, dtype, devices)` therefore races every slot of every layer,
  streamed ones included, outside any checkpoint on random scratch values of the
  factors (restored afterwards), over every token bucket and grad pattern; the training
  build runs it before step 0 unless `TUNE="off"`. IA3 and GLoRA have no tier op: IA3
  runs the device's preferred fused backend, GLoRA merge its fused merge kernel and
  GLoRA bypass the eager body.
- a backend name (`triton` | `tilelang` | `compile` | `torch`): every op calls that
  backend directly, with no measurement; a name the device cannot run is an error.

The base output always comes from `MossLinear._impl`, and the base weight from the
source each Target selects once (dense parameter, quantized / streamed storage, or an
override). Attaching, merging or removing adapters re-binds the owning backbone's ops
(`bind_ops()`), so a fused linear+epilogue kernel never reads a weight past an adapter.
Changing the multiplier (`set_multiplier`) or the adapter state (`load_state_dict`)
re-binds and builds fresh slots; factor dtypes and `requires_grad` flags are part of a
slot's bound state, so changing them calls for `AdapterSet.bind()`.

## Steps

The training loop calls `begin_step(sigma, seq, step)` before every forward (each
microbatch, validation included) and `end_step()` after the step's last backward:

- T-LoRA masks: one per token axis. Joint-block layers read a row per joint token,
  text-refiner layers a row per text token, the video embedder / final linear a row
  per image token, modulation layers a row per sample (`adapters.tlora.AXES`, set per
  layer from its path). `end_step` clears them, so sampling and exports see the
  full-rank adapter, which is what the exported files hold.
- Random draws: each layer's DyLoRA rank prefix and module-dropout skip come from a
  generator seeded by (`SEED`, step, microbatch index), drawn once outside the forward:
  a checkpoint recompute replays them, every data-parallel rank draws the same ones,
  and a resumed run repeats them. Without `begin_step` DyLoRA uses every block and no
  layer is skipped.

## Optimiser groups

`param_groups(lr, weight_decay, rules, use_muon)` follows `WD_RULES` (`moss`: no decay
on 1-D tensors, DoRA magnitudes, rescales, scalars, T-LoRA's lambda; `none`; `all`) and
marks 2-D factors with both sides > 1 Muon-eligible under `OPTIMIZER="muon"`. LoRA+
(`LORAPLUS_RATIO`) multiplies the learning rate of each kind's zero-initialised
factors: LoCon `lora_up`, LoHa `hada_w2_a`, LoKr `lokr_w2` / `lokr_w2_b`, GLoRA `a2` /
`b2`, DyLoRA's up blocks; T-LoRA's `p_layer` (data-initialised, its output side).

## Quantized and streamed bases

When a layer's base lives in storage (`MossLinear.set_base`), its dense parameter is
empty and every adapter path reads the base through the storage; the base matmul stays
the storage's own `linear`. Such layers always bind the bypass form, since nothing can
be merged into the storage; `merge_into` refuses them, `ALGO="full"` is rejected, and
merged exports dequantize the storage.

## Multiplier

The multiplier scales dW. For DoRA it moves the decomposed weight linearly toward the
base weight, `W + m*(W' - W)`, which is how ComfyUI applies strength; for OFT/BOFT it
interpolates each rotation toward identity, `m*R + (1 - m)*I` (LyCORIS's functional
`diff_weight(..., scale=m)`).

## State and export

- `state_dict()` / `save_state()` hold the raw parameters by LyCORIS key (exact resume;
  the file also records the spec); `load_state_dict` requires exactly the live keys.
- `export(path, fmt)` writes `lycoris` (LyCORIS's own state dicts; a layer with a
  per-row fold scale is written as `diff`), `comfy` (see
  `docs/guides/export-to-comfy.md`), `merged` (one bare-key transformer file) or
  `diffusers` (the transformer folder: shards, index and `config.json`). The merged
  formats go through the backbone's own export state
  (`kohakumoss.model.io.save.export_state`), which dequantizes stored bases and undoes
  the residual fold; `row_scales` undoes the fold for `lycoris` / `comfy`.
- Every export and `merge_into` evaluates the adapters outside training (no rank
  dropout, every DyLoRA block, no T-LoRA mask) and leaves the training state as it was.
- `merge_into(m)` bakes every linear and norm delta at `m`.

## Inference

`load_adapters(model, [(file, multiplier), ...], mode)` reads LyCORIS files,
ComfyUI-style files (`lora_unet_` or `diffusion_model.` keys) and training states (their
spec rebuilt without dropout, on the device's own tier):

- `bypass` stacks each adapter's forward on the previous one. Low-rank paths stay
  factorised; DoRA, dense diffs, DyLoRA, `full` and OFT at multiplier != 1 apply a dW
  held once in the layer's compute dtype, computed against the layer's current
  effective weight (the base plus every adapter stacked before it), as `merge` does;
- `merge` adds dW into the base weights, summed in fp32 and rounded once.

Loaded modules run on the device's preferred tier. `remove()` restores every touched
weight and forward bit-exactly; `undoable=False` keeps no copies (a one-way merge, as
`scripts/export/merge.py` does). `row_scales` places unfolded files onto a folded
model.
