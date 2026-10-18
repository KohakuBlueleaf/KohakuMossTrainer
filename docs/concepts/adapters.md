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

`ALGO="full"` is the full finetune ([full-finetune.md](../guides/full-finetune.md)): the
backbone's own weights train and no LyCORIS module is built.

`ALGO_KWARGS` keys are checked against the LyCORIS module's constructor (and
`LycorisNetwork`'s): an unknown key, or one with its own knob (`alpha`, `lora_dim`,
the dropouts), is an error. T-LoRA's mask takes `mask_min_rank`, `mask_alpha` and
`mask_max_timestep` (LyCORIS's `compute_timestep_mask` with t = sigma * max_timestep);
these never reach the module. DoRA rejects input dropout (LyCORIS ignores it).

`TRAIN_NORMS=True` adds a trained delta to the norms the preset covers
(`adapters.train.norms.NormDelta`, a parametrization: `weight = original + m * w_norm`,
summed in fp32 and rounded once to the norm's dtype).

Each algorithm is one `adapters.kinds.Kind` record (dW builders, fused and eager
forward factories, ComfyUI conversion, row-scale tensors, inference stacking class,
LoRA+ side), registered by kind name, LyCORIS module class and `ALGO` name;
`adapters.register(kind)` adds one. See `docs/internals/adapter-kinds.md`.

## Targeting

`PRESET` is resolved by `adapters.build.presets.resolve_preset` into a full LyCORIS preset
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

On a model whose bases live in quantized or streamed storage, no dense base weight is
built: LyCORIS sizes each module from the linear's features, and the construction state
that reads the base (the DoRA magnitude, the T-LoRA SVD, the OFT / BOFT rescale) is taken
from one transient copy of each layer's base, one layer at a time.

## Binding

`apply(model, config)` (an `AdapterConfig`, from the knobs by `config_from_knobs`) builds LyCORIS modules on the matched linears (the target layers
are `MossLinear`, an `nn.Linear`), registers each as the child `lyco` of its linear
(FSDP sharding and weight streaming see it with its layer), and installs each layer's
forward once. Per layer the bound forward is one of:

- the kind's kernel ops: `bypass` adds the adapter path to the base output, `merge`
  builds dW and runs one GEMM on W + dW, summed in fp32 and rounded once;
- the DoRA path for decomposed algorithms: W' = DoRA(W + dW), the decompose step being
  a weight-decompose kernel op;
- the LyCORIS module's methods: while training with input or rank dropout (the kernel
  ops carry neither), while training DoRA with `DORA_DETACH`, for GLoRA bypass, OFT at
  multiplier != 1, DyLoRA, T-LoRA, and (named fused backend only) where the layer is
  outside that kernel's scope (LoHa rank above `lycoris.kernels.limits.MAX_HADA_RANK`,
  LoKr factors above the kernel bound, an OFT block above the backend's bound).

Each kernel op pairs the LyCORIS fused op (`lycoris.kernels.autograd`, by `backend=`)
with its eager formula (the public `lycoris.functional` entry on backend `"torch"`).
Forwards that differ between train() and eval() (dropout, DoRA detach, DyLoRA, module
dropout) re-bind on the mode change, so sampling and validation take the kernel ops.
The base add goes inside the op where one takes it
(docs/internals/adapter-binding.md#forward-wrappers). How a kernel op runs depends on
`KERNELS.lycoris`:

- `auto` (default): on a device with a fused LyCORIS backend each op is an
  `AutotunedOp`: the fused kernel of the device's most-prior backend against
  `torch.compile` of the eager formula, measured once per problem bucket by the kernel
  tuner (forward + backward on detached copies of the real operands), stored per card
  and bound (docs/internals/adapter-binding.md#autotuned-ops). A call inside a
  checkpointed forward or a backward runs the kernel unsettled, so a checkpoint
  recompute reruns what its forward ran. `AdapterSet.prewarm(rows, dtype, devices)`
  therefore settles every distinct op, streamed layers included, outside any
  checkpoint on scratch inputs over every token bucket and grad pattern (the factors
  keep their values and no gradient); the training build runs it before step 0 unless
  `TUNE="off"`. Precision against PyTorch is checked by the test suite
  (`tests/adapters/test_kernel_ops.py`), not in the measurement. Without a fused
  backend (CPU, XPU) every op runs the eager formula.
- a backend name (`triton` | `tilelang` | `compile` | `torch`): every op runs that
  backend (fused kernel, compiled formula, eager formula), with no measurement; a name
  the device cannot run is an error.

The base output always comes from `MossLinear._impl`, and the base weight from the
source each Target selects once (dense parameter, quantized / streamed storage, or an
override). Attaching, merging or removing adapters re-binds the owning backbone's ops
(`bind_ops()`), so a fused linear+epilogue kernel never reads a weight past an adapter.
Changing the multiplier (`set_multiplier`) or the adapter state (`load_state_dict`)
re-binds every layer; factor dtypes and `requires_grad` flags are part of a layer's
bound forward, so changing them calls for `AdapterSet.bind()`.

## Steps

The training loop calls `begin_step(sigma, seq, step)` before every forward (each
microbatch, validation included) and `end_step()` after the step's last backward:

- T-LoRA masks: one per token axis. Joint-block layers read a row per joint token,
  text-refiner layers a row per text token, the video embedder / final linear a row
  per image token, modulation layers a row per sample (`adapters.train.tlora.AXES`, set per
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

When a layer's base is not a plain 16/32-bit dense parameter (quantized or streamed
storage via `MossLinear.set_base`, or fp8-compute copies), its dense parameter is an
empty placeholder and the base matmul stays the storage's own `linear`. Such layers bind
the bypass form only: the adapter's delta is added to the base output, no dense weight
or dW is held, and `ADAPTER_MODE="merge"` is a validation error naming the base. DoRA
reads one transient copy of the base per step for the norm of W + dW
(docs/internals/adapter-binding.md#quantized-and-streamed-bases). `merge_into` refuses
such layers and merged exports dequantize the storage one layer at a time.

## Multiplier

The multiplier scales dW. For DoRA it moves the decomposed weight linearly toward the
base weight, `W + m*(W' - W)` with `W' = decompose(W + dW)`, which is how ComfyUI applies
strength; for OFT/BOFT it interpolates each rotation toward identity,
`m*R + (1 - m)*I` (LyCORIS's functional `diff_weight(..., scale=m)`).

LyCORIS's own DoRA runtime (`weight_decompose(W + dW, dora_scale, m)`, used by
`LoConModule.forward` / `get_merged_weight` and so by the kohya and sd-webui LyCORIS
loaders) scales the magnitude instead: `(W + dW) * (m * (s - 1) + 1)` with `s` the
per-row (or per-column) ratio `dora_scale / ||W + dW||`. The two agree at `m = 1` and
differ at any other strength, so a DoRA file at strength 0.7 renders differently there
than in KohakuMoss and ComfyUI. KohakuMoss follows ComfyUI because its exports are made
for ComfyUI (the `comfy` format) and because a linear move keeps every format's
strength one rule (`diff`-written layers, merges and the DoRA kernels all compute
`W + m * delta`).

## State and export

- `state_dict()` / `save_state()` hold the raw parameters by LyCORIS key (exact resume);
  `state_metadata()` is the file's metadata, written by `save_state` and by every
  training checkpoint's `trainable.safetensors`: the raw-state tag, the adapter
  configuration (`kohakumoss_spec`) and the weight fold the layers trained on
  (`kohakumoss_fold`, a `fold_record`: per-block scales, via, pre-gate scales, eps; null
  when unfolded). A full finetune's checkpoint carries its own tag and the fold.
  `load_state_dict` requires exactly the live keys.
- `export(path, fmt)` writes `lycoris` (LyCORIS's own state dicts; a layer with a
  per-row fold scale is written as `diff`), `comfy` (see
  `docs/guides/export-to-comfy.md`), `merged` (one bare-key transformer file) or
  `diffusers` (the transformer folder: shards, index and `config.json`). The merged
  formats go through the backbone's own export state
  (`kohakumoss.model.io.save.export_state`), which dequantizes stored bases and undoes
  the residual fold; `row_scales` undoes the fold for `lycoris` / `comfy`;
  `fused_qkv` writes ComfyUI's fused `proj.qkv` layout for `comfy` and `merged`
  (`docs/guides/export-to-comfy.md#fused-qkv`).
- Every export and `merge_into` evaluates the adapters outside training (no rank
  dropout, every DyLoRA block, no T-LoRA mask) and leaves the training state as it was.
- `merge_into(m)` bakes every linear and norm delta at `m`.

## Inference

`load_adapters(model, [(file, multiplier), ...], mode)` reads LyCORIS files,
ComfyUI-style files (`lora_unet_` or `diffusion_model.` keys; a fused `proj.qkv` LoRA or
`diff` layer is split into q / k / v) and training states. A training state (tagged
with its configuration in the file's metadata) is converted first
to the unfolded LyCORIS tensors an export writes: its modules rebuilt from its
configuration without dropout, on the device's own backend, each layer's dW rows
divided by the fold the state trained on (its recorded fold; this model's own when it
records none), the base a layer reads during that conversion being the effective
weight in the state's own space. It then loads like a LyCORIS file, so a state trained
on a folded model renders correctly on an unfolded one and the reverse:

- `bypass` stacks each adapter's forward on the previous one. On a dense base low-rank
  paths stay factorised; DoRA, dense diffs, DyLoRA and OFT at multiplier != 1 apply a
  dW held once in the layer's compute dtype, computed against the layer's current
  effective weight (the base plus every adapter stacked before it), as `merge` does;
- `merge` adds dW into the base weights, summed in fp32 and rounded once;
- on a quantized / streamed base every kind attaches as a bypass chain whatever the
  mode (a merge request falls back with one log line): DoRA keeps only its per-row (or
  per-column) ratio, computed once from a transient copy of the effective weight, and a
  dense-diff file is refused.

Loaded modules run their own methods (LyCORIS picks the backend per call). `remove()`
restores every touched
weight and forward bit-exactly; `undoable=False` keeps no copies (a one-way merge, as
`scripts/export/merge.py` does). `row_scales` places unfolded files onto a folded
model.
