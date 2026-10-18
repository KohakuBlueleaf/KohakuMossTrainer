# Configs and presets

Every script is a kogine script: its settings are module-level UPPER_CASE knobs, and a
config file is a Python file of the same names. `kogine run <script> --config <file>
--set KEY=VALUE` loads the file, applies each `--set` (coerced to the knob's type) and
calls the script's `main()`. Configs compose with `use_config("<file>")`: the named file's
knobs apply first, later calls override earlier ones, and the file's own assignments
override all of them. `use_config` replaces whole values; it does not merge dicts.

## From a base config

```bash
kohakumoss-config new --from lora_24g_bf16 my_run.py
kohakumoss-config wizard
```

`new --from <base> <file>` writes a config that extends a base (a recipe name from
`kohakumoss-config list`, or any train config path) with
`use_config("<base>")`, so later changes to the base flow through. Below it the file
lists every knob of `scripts/train/moss.py`, in the script's sections and order, each
with the script's one-line comment: `WEIGHTS`, `DATASETS`, `RUN_NAME` and `STEPS` are
active at the base's values, every other knob is commented out at the value the base
gives it. Edit `WEIGHTS` / `DATASETS`, uncomment what you change, and run
`kogine run scripts/train/moss.py --config my_run.py`.

`wizard` asks for GPU memory (default: device 0's), precision (default: bf16 where the
arch has it), algorithm, dataset layout and path, weights path, steps, GPUs, run name and
file, then writes the same commented file from the preset layers below, plus the
data-preparation config.

## Starter configs from flags

```bash
kohakumoss-config new train --weights path/to/MOSS-Image-Base --data path/to/dataset
kohakumoss-config new sample --weights path/to/MOSS-Image-Base --adapter runs/moss/my-lora/export/my-lora-comfy.safetensors
kohakumoss-config list
kohakumoss-config check configs/user/my-lora.py
```

`new train` writes two files into `configs/user/` of a clone (git-ignored), or the
current directory for a PyPI install (`--out` puts them elsewhere):

- `<name>.py`, the train config: a `use_config` line per preset layer, then every knob
  as in `new --from`, with `WEIGHTS`, `DATASETS` (the data layer's entry with your
  path), `RUN_NAME` and anything from `--steps` / `--prompt` / `--gpus` active.
- `<name>_data.py`, the data-preparation config: `use_config("<name>.py")` plus the
  cache encoders (`kohakumoss.encoders.text_encoder`, `kohakumoss.encoders.vae_encoder`
  on the weights). The data scripts read the same data knobs as training, so the cache
  keys match by construction; the encoder knobs stay out of the train config, whose
  knob check rejects names the train script does not declare.

Both docstrings list the commands to run, in order (inspect, text cache, latent cache,
train). Flags left out are read from the local GPU: `--vram` from device 0's memory in
exact GiB, `--precision` bf16 where the arch has it, else fp16. A card selects the
largest VRAM layer it holds within `--vram-tolerance`, a fraction of the layer's size
(default 0.05): a 24 GB card reports about 23.99 GiB and gets `vram_24g`; with
`--vram-tolerance 0` it gets the next smaller layer. The sample starter picks its DiT
placement with the same rule. `--gpus N` (Linux) adds
the parallel layer. The composed knobs pass through the train script's knob check
before anything is written; an invalid combination is refused with every problem
listed, and a path that does not exist yet is only a warning.

`new sample` writes `<name>.py` (default `sample`) for `scripts/sample/sample.py`.

## Layers

A layer sets one concern and is composed after the base. Train layers live in
`configs/moss/<group>/`, sample layers in `configs/sample/<group>/`, in this order:

| group | layers | sets |
| --- | --- | --- |
| `_base` | `base_lora` (`base_full` = base_lora + algo_full) | weights / data placeholders, objective, schedule, caches, sampling |
| `algo` | `lora`, `loha`, `lokr`, `oft`, `dora`, `full` | ALGO, DIM, ALPHA, ALGO_KWARGS, DROPOUT, LR; `full`: optimiser, EMA, exports, `ckpt-resident` |
| `data` | `kohya`, `folder`, `folder_nl`, `jsonl`, `parquet`, `webdataset`, `moss_shard` | the DATASETS entry template with every source argument, CAPTION, ASPECT_RANGE |
| `vram` | `6g`, `8g`, `12g`, `16g`, `24g`, `auto`, `24g_full` | MEMORY_PROFILE, MAX_TOKENS, TEXT_ENCODE, VAE_ENCODE; `6g`: the offline caches; `6g` / `8g`: SAVE_MINUTES 30; `24g_full`: AdamW without EMA |
| `precision` | `bf16`, `fp16` | PRECISION, FP16_FIX (fp16: `online`, c = 1/16) |
| `parallel` | `ddp`, `fsdp`, `pipeline`, `threaded` | PARALLEL, DEVICES and what the engine needs (see [multi-gpu.md](multi-gpu.md)) |

### Algorithm arguments

`ALGO_KWARGS` goes to the LyCORIS module constructor unchanged, under LyCORIS's own
argument names; the knob check rejects a name the module does not take. The algorithm
layers spell out the main ones at LyCORIS's defaults (except where the layer changes
them), so a config edits them in place:

| algo | ALGO_KWARGS | |
| --- | --- | --- |
| LoRA / LoCon (`lora`), LoHa (`loha`) | `use_tucker`, `use_scalar`, `rank_dropout_scale` | Tucker for conv, a trainable scalar, rescale after rank dropout |
| LoKr (`lokr`) | `factor` (-1 = the near-square split; LyCORIS default), `full_matrix`, `decompose_both`, `use_tucker`, `rank_dropout_scale` | `full_matrix=True` keeps w1 / w2 full at any DIM; DIM >= half the larger factor also does |
| OFT (`oft`) | `constraint`, `rescaled` | norm constraint of the rotation, trainable rescale |

Dropout is its own knob, `DROPOUT = {"input": ..., "rank": ..., "module": ...}`, mapped to
LyCORIS's `dropout`, `rank_dropout` and `module_dropout` (those names are not accepted in
ALGO_KWARGS). DoRA is `DORA = True` (LyCORIS `weight_decompose`).

The data layer comes before the VRAM layer so the VRAM layer owns MAX_TOKENS; the VRAM
layer comes after the algorithm layer because `24g_full` replaces the optimiser
(trainable weight streaming runs AdamW).

VRAM choice: the largest adapter layer the GPU holds within `--vram-tolerance` (a card
of at least `size x (1 - tolerance)` GiB, 22.8 GiB for `24g` at the default 0.05);
above 24 GiB, `auto` (the planner takes the fastest profile whose estimate fits, see
[profiles](../concepts/profiles.md)). Full finetune has `24g_full` and `auto` above
24 GiB; a card that does not hold 24 GiB within the tolerance is refused. Below the
smallest layer (6 GiB) the starter is refused; the message names `--vram` and
`--vram-tolerance`. Each adapter layer is a catalog profile:

| layer | MEMORY_PROFILE | MAX_TOKENS | text encoder / VAE |
| --- | --- | --- | --- |
| `24g` | `balanced` | 12288 | TE stream, 8 batches per pass |
| `16g` | `low` | 8192 | TE stream, 4 batches per pass |
| `12g` | `ckpt-stream4-actoff-optstream-gradoff` | 6144 | TE stream; VAE offload |
| `8g` | `minimal` | 4352 | TE stream, 8 batches per pass; VAE offload, tiled |
| `6g` | `ckpt-stream2-actoff-optcpu-gradoff` | 4352 | offline latent and text caches (the base's TE plan for sample prompts) |

The 8 GB and 6 GB targets are checked by the bench matrix's `cap` runs (the allocator
capped below the target, pass when the nvidia-smi peak stays at or below it), and the
step time and nvidia-smi peak of each layer per card are in the bench matrix's speed
table (`report.md`, [bench-matrix.md](../performance/bench-matrix.md#report)). A VRAM layer never
sets precision: a quantized base is the `BASE_PRECISION` knob (the 12 GB and XPU
recipes set `int8`).

The VRAM layers from `8g` up set the text-encoder plan `TEXT_ENCODE`
([text-encoder.md](../internals/text-encoder.md)); `6g` keeps the base's. The trainer loads the text encoder
whenever training-time sampling or an online text cache is on, and the encoder's
15.63 GiB (bf16) does not fit next to the 9.47 GiB transformer on cards up to 24 GiB,
so those layers stream it through a layer ring (`memory="stream"`, about 2 GiB during a
pass and nothing between passes) and encode `ahead` batches
per pass. With an offline text cache the encoder runs once for the sample prompts and
is freed. `auto` picks the fastest placement that fits next to the DiT step. The CPU
is for the offline precompute pass (`scripts/data/cache_text.py`) only.

### Host RAM

What a single-GPU run holds in host memory, from the component sizes (bf16 base; an
int8 / fp8 base halves the DiT parts):

| part | when | GiB |
| --- | --- | --- |
| trainer process and its `NUM_WORKERS` loader workers (spawned processes on Windows, each importing torch) | always | the bench matrix budgets 8 per run (`kohakumoss_bench/plan/hostmem.py`) |
| the DiT, loaded on the host before placement | build | 9.47 |
| the text encoder, read into host RAM | build, when `SAMPLE_EVERY > 0` or `TEXT_CACHE` is not `offline`; with an offline text cache it is freed after the sample prompts | 15.63 |
| streamed DiT blocks, pinned (0.21 per block of 42; the device keeps `stream_resident`) | streaming profiles | about 5.5 (`balanced`) to 8.4 (two resident blocks) |
| offloaded activations (block inputs, one per block, in pinned buffers rounded up to a power of two) | `act_offload` | about 1 at 4352 tokens, 2 at 8192 |
| adapter master, moments and gradient sums (16 bytes per trainable parameter) | `optim_state` `stream` / `cpu`, `grad_offload` | about 1.1 for LoRA DIM 32 on `attn-mlp` (74 M parameters) |

So the `8g` layer peaks at build near 32 GiB (process, streamed blocks and the text
encoder) and holds about 19 GiB during the steps; `24g` (`balanced`) peaks near 29 GiB
and holds about 14 GiB. With `SAMPLE_EVERY = 0` and both caches offline the text
encoder never loads, which removes the 15.63 GiB build peak. Full finetune:
[full-finetune.md](full-finetune.md#host-ram).

Multi-GPU: `kohakumoss-config new --gpus N` gives adapters `ddp`. Full finetune gets
`fsdp` when the per-rank FSDP2 footprint (the `fsdp` layer's profile, everything sharded
over N ranks, by the memory estimate) fits the card within `--vram-tolerance`, else
`ddp` over the full VRAM layer's swap profile; `--vram auto` gives `ddp`
([full-finetune.md](full-finetune.md)). `pipeline` and `threaded` are composed by hand. `ddp` keeps the VRAM layer's
profile unchanged (every profile runs under DDP); `fsdp`, `pipeline` and `threaded`
replace the profile with
`ckpt-resident` (resident compute-precision weights, GPU states) and turn sampling off.
Since `use_config` replaces the whole dict, `kohakumoss-config new` writes
MEMORY_OVERRIDES merged from every layer that sets it, in layer order, into the starter;
a hand-written config that stacks two such layers merges the dicts itself.

Sample layers:

| group | layers | sets |
| --- | --- | --- |
| `_base` | `base_sample` | the official sampler (50 steps, CFG 3, shift 5), placeholders |
| `precision` | `bf16`, `fp16` | PRECISION, FP16_FIX (fp16: `online`, c = 1/16) |

The sample knobs carry the train script's names and values (`PRECISION`,
`BASE_PRECISION`, `MEMORY_PROFILE`, `TEXT_ENCODE`, `VAE_ENCODE`, `VARIANT`).
`new sample` allows every card: it writes the `MEMORY_PROFILE` of the card's training
VRAM layer (the `6g` layer's below 6 GiB; no profile, the whole DiT resident, above the
largest layer), so sampling holds the DiT the way training does on that card, and
leaves `TEXT_ENCODE` / `VAE_ENCODE` unset: both are `auto`, the fastest placement that
fits next to the DiT ([sampling-memory.md](../internals/sampling-memory.md)). The base
weights are 9.47 GiB (transformer, bf16), 15.63 GiB (text encoder, bf16) and 0.23 GiB
(VAE).

## Recipes

`configs/moss/recipes/` are fixed compositions of the same layers (plus a RUN_NAME and
any recipe-specific value). Run one directly, `use_config` it from your own file, or
extend it with `kohakumoss-config new --from <name>`:

| recipe | composition |
| --- | --- |
| `lora_24g_bf16` | base_lora + vram_24g + precision_bf16 |
| `lora_16g_fp16` | base_lora + vram_16g + precision_fp16 (V100, 2080 Ti) |
| `lora_12g_stream` | base_lora + vram_12g, `BASE_PRECISION="int8"` |
| `lora_xpu_b50` | base_lora + vram_16g + precision_bf16, `BASE_PRECISION="int8"`, 6144 tokens (Intel Arc Pro B50) |
| `lora_ddp` | base_lora + vram_24g + precision_bf16 + parallel_ddp (two GPUs, Linux) |
| `dora` | base_lora + algo_dora + vram_24g + precision_bf16 |
| `lycoris_loha`, `lycoris_lokr`, `lycoris_oft` | base_lora + algo_loha / algo_lokr / algo_oft + vram_24g + precision_bf16 |
| `full_24g_offload` | base_full + vram_24g_full (one 24 GB GPU) |
| `full_fsdp_4xv100` | base_full + precision_fp16 + parallel_fsdp, four GPUs, 8192 tokens |

The other groups under `configs/`: `configs/moss/quality/` (below),
`configs/moss/scale/` (the multi-GPU scaling layers, [multi-gpu.md](multi-gpu.md)),
`configs/data/` (a data-preparation config for the `scripts/data/` passes,
[datasets.md](datasets.md)) and `configs/sample/` (the sample layers above).

## Quality-test configs

`configs/moss/quality/quality_lora8.py` (LoRA DIM 8, ALPHA 8, LR 5e-4) and
`quality_lokr16.py` (LoKr factor 16, full matrix, DIM 10000, ALPHA 1, LR 2e-3) train 400
steps with AdamW at a constant LR after 20 warmup steps, seed 0, a checkpoint every 100
steps (all kept), the online text cache with the streamed text encoder and the offline
latent cache, on the `auto` VRAM layer. They are bf16; fp16 is either the online fix,
`--set PRECISION=fp16 --set FP16_FIX=online --set FP16_FIX_KWARGS={"c":0.0625}`, or the
fp16-fix checkpoint: `--set PRECISION=fp16 --set WEIGHTS=<the fp16fix .safetensors>
--set TEXT_ENCODER=<base>/text_encoder --set VAE=<base>/vae` with `FP16_FIX` left at
`"none"`. The trainer reads the fold from the file's `kohakumoss_fp16_fix` metadata,
marks the model folded, and exports adapters with the fold undone (they load on the
unfixed base). A single-file `WEIGHTS` holds only the transformer, so the knob check
asks for `VAE` / `TEXT_ENCODER` whenever the run loads them.
`scripts/sample/compare_grid.py` renders the result: rows = prompts, columns = the base
model then each checkpoint ([inference.md](inference.md)).

## Validation

Every config under `configs/` names the script it drives in its docstring
(`scripts/<...>.py`, the first one wins), and `tests/configs/test_shipped_configs.py`
checks each one through that script's knob checks: the train script's
`kohakumoss.training.knobs.validate`, the sample script's
`kohakumoss.inference.knobs.validate`, and `kohakumoss_data.data_problems` for the data
scripts. A layer is checked on top of its script's base (`base_lora` / `base_sample`), or
on the base it names in a private `_BASE = "<path>"` (kogine ignores names starting with
`_`). `kohakumoss-config check <file>...` runs the same check on any config.
