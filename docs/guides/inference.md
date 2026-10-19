# Inference

`kohakumoss.inference.MossPipeline` generates images with the official MOSS sampler:

- Euler over the shift-5 sigma grid, snapped to the 1000-step training grid;
- CFG 3, with the negative branch skipping block 22.

`scripts/sample/sample.py` drives it from the command line ([CLI](#cli));
`scripts/sample/compare_grid.py` renders checkpoints side by side
([Comparison grid](#comparison-grid)); the Python API takes the same settings
([Python](#python)).

## CLI

`scripts/sample/sample.py` is a kogine script: every setting is a module-level knob, set
in a config file or with `--set`. `WEIGHTS` is the ComfyUI models folder the README
downloads (`ComfyUI/models`; `TRANSFORMER`, `TEXT_ENCODER`, `VAE` name its files, the
int8 DiT and text encoder by default) or an HF pipeline folder (`models/MOSS-Image-Base`,
with `transformer/`, `text_encoder/`, `vae/`); see [weights.md](weights.md):

```bash
kogine run scripts/sample/sample.py --set WEIGHTS=ComfyUI/models \
    --set PROMPT="a cat" --set ADAPTERS=my_lora.safetensors:0.8 --set OUT=out/samples
```

A config file holds the same knobs:

```python
WEIGHTS = "ComfyUI/models"
PROMPTS = ["a cat", "a dog"]
SIZES = [(1024, 1024), (832, 1216)]  # (H, W) or "HxW" per prompt; the last repeats
SEEDS = [0, 1]  # missing seeds are SEED_BASE + index
ADAPTERS = [("my_lora.safetensors", 0.8)]
OUT = "out/samples"
```

`kogine run scripts/sample/sample.py --config my_sample.py --set STEPS=30`

Two ways to get such a config without writing it:

- `kohakumoss-config new sample --weights <dir> --adapter <file>` writes one for the
  local GPU: the precision layer and the `MEMORY_PROFILE` of the card's training VRAM
  layer, with the text encoder and VAE left at `auto` ([configs.md](configs.md)). Every
  card gets one.
- `scripts/examples/sample.ps1` / `sample.sh` and `grid.ps1` / `grid.sh` write and check
  one from a few options and run it
  ([scripts/examples/README.md](../../scripts/examples/README.md)).

### Knobs

- `PROMPT` is one prompt; `PROMPTS` is a list. A `--set PROMPTS=` value is a list in
  list syntax (`--set 'PROMPTS=["a, b", "c"]'`, or `[a, b]` split at its commas); a bare
  string is one prompt, its commas kept (same for compare_grid.py's `PROMPTS`).
- `PROMPTS_FILE` takes a `.txt` (one prompt per line) or a `.json` list of strings or of
  `{"prompt", "negative", "size": [H, W], "seed"}` objects. Items come in the order
  `PROMPTS`, `PROMPT`, then the file.
- `PRECISION = "fp16"` with `FP16_FIX = "online"` and `FP16_FIX_KWARGS = {"c": 0.0625}`
  selects an fp16 fix; `PRECISION = "auto"` is bf16 where the card has it, else fp16.
- `VARIANT` is the arch preset, `"base"` (default) or `"sft"`, as in training; on a root
  with `base/` / `sft/` folders `WEIGHTS/<VARIANT>/` is read, and a pipeline folder is
  read as is ([Weights root](#weights-root)).
- Every knob is checked before any weight loads (`kohakumoss.inference.knobs.validate`);
  all problems are reported in one error.

### Shared with the train script

The knobs are named and valued as the train script's (`PRECISION`, `BASE_PRECISION`,
`MEMORY_PROFILE`, `TEXT_ENCODE`, `VAE_ENCODE`, `VARIANT`, ...), so a value copies
between the two, except:

| value | valid in |
| --- | --- |
| `MEMORY_PROFILE="manual"` with the training-only knobs | train |
| `MEMORY_PROFILE=""` (the whole DiT resident) | sample |
| `PRECISION="fp32"` | sample |
| `MEMORY_PROFILE="auto"` | both; in sampling it is the fastest profile whose DiT forward of one 1024x1024 CFG sample (the inference estimate) fits the free device memory, logged as `MEMORY_PROFILE 'auto' -> <profile>`; the text encoder and VAE then place themselves around it |

The train script's sample knobs map to this script's:

| train script | sample.py |
| --- | --- |
| `SAMPLE_PROMPTS` | `PROMPTS` |
| `SAMPLE_NEGATIVE` | `NEGATIVE` |
| `SAMPLE_SIZE` | `SIZES` |
| `SAMPLE_SEEDS` | `SEEDS` |
| `SAMPLE_SKIP_LAYERS` | `SKIP_LAYERS` |
| `SAMPLE_STEPS` | `STEPS` |
| `SAMPLE_CFG` | `CFG` |
| `SAMPLE_SHIFT` | `SHIFT` |

### Output

- One PNG per prompt, numbered after the highest `NNNN_` index already in `OUT`.
- One `metadata-<first index>.json` per run with every knob and the resolved pipeline
  settings; reusing `OUT` keeps earlier images.
- The run logs every phase with its time and shows a progress bar per batch
  ([logging.md](logging.md)).

## Python

```python
import torch
from kohakumoss.inference import MossPipeline

pipe = MossPipeline.from_pretrained(
    "ComfyUI/models",            # a ComfyUI models folder, a pipeline dir, or a root with base/ (and sft/)
    dtype=torch.bfloat16,
    adapters=[("my_lora.safetensors", 0.8)],
    adapter_mode="merge",        # or "bypass"
    text_plan={"memory": "auto"},  # TE: "auto" | "resident" | "offload" | "ring" | "cpu"
    vae_plan={"memory": "auto"},   # VAE: "auto" | "resident" | "offload" | "ring" | "cpu"
)
out = pipe.generate(
    ["a cat on a windowsill", "a lighthouse at dusk"],
    sizes=[(1024, 1024), (832, 1216)],
    seeds=[0, 1],
    steps=50, cfg=3.0, shift=5.0,
)
out.images[0].save("cat.png")
```

The settings below are `from_pretrained` / `generate` arguments; the knob names of
`scripts/sample/sample.py` are in parentheses where they differ.

### Sampling

- **Packed CFG** (`cfg_mode="packed"`, default): the conditional and unconditional samples
  of a batch run in one packed forward; `two_pass` runs them as two forwards with half
  the tokens each.
- **Per-sample determinism**: sample i's noise and conditioning depend only on its
  prompt, size and seed: no other sample's content reaches it, and its slot in the batch
  does not matter (packed attention and the text packs keep samples apart). Bitwise, a
  sample also depends on the batch's shapes, the other samples' sizes and token counts,
  which set the GEMM, attention and text-encoder problem shapes and therefore their
  rounding; the same batch composition reproduces a sample bit for bit.
- **Embeddings in**: `prompt_embeds` skips encoding the prompts; `negative_prompts`
  strings are still encoded unless `negative_embeds` is given.
- **Linear algebra**: an owned CUDA pipeline selects cuSOLVER (OFT's batched Cayley
  inverse) for the duration of each `generate` call and restores the previous backend
  after it.

### Weights root

`path` is a ComfyUI models folder: `transformer`, `text_encoder_file`, `vae_file`
(`TRANSFORMER`, `TEXT_ENCODER`, `VAE`) name its files, each "" for its default
(`moss_image_<variant>_int8_convrot`, `qwen3.5_9b_int8_convrot`, `flux2-vae_small`, each
falling back to the next file that exists; [weights.md](weights.md#comfyui-layout)).

Otherwise `path` is one pipeline folder, or a root with `base/` (and `sft/`)
subfolders. `variant` (`VARIANT`, `"base"` default or `"sft"`) is the arch preset:

- Each component comes from `path/variant` when that folder holds its weights (a
  component folder with safetensors files), else from `path/base` (logged). An SFT
  folder that carries only the transformer weights (its `text_encoder/` and `vae/` with
  configs and no weights) takes the base's text encoder and VAE.
- Without a `path/variant` folder, `path` itself is the pipeline folder.
- A root that has no folder for the requested variant (e.g. `sft` on a root with only
  `base/`) is an error.
- The folders used are in `out.settings["sources"]`.

### DiT memory plan

`memory=` a profile (`MEMORY_PROFILE`, `MEMORY_OVERRIDES`; any catalog entry,
[profiles.md](../concepts/profiles.md)) places the DiT's block weights:

| profile | DiT blocks |
| --- | --- |
| `speed` (or no profile) | the whole DiT resident |
| `balanced` | streamed through a ring, 16 resident |
| `low` | streamed, 4 resident |
| `minimal` | streamed, 2 resident |
| `manual` | `STREAM` / `STREAM_RESIDENT` / `PINNED` |

- Inference applies the weight placement of a profile. The base precision is its own knob (`BASE_PRECISION`,
  [fp8 / MXFP8 compute](#fp8--mxfp8-compute)) under every profile.
- Adapters attach after the plan placed the DiT. On a quantized or streamed base a merge
  runs as bypass with a WARNING (a merge needs a 16/32-bit dense base), and the adapter
  log line names the effective mode (`merge + bypass (N layers on stored bases)`).
- Each call runs under `inference_mode`, and the allocator cache is returned to the
  driver after the text encode, the sampling and the decode.

Per-phase memory (fp16, 1024^2, TE ring fp8, RTX 3090; `scripts/bench/infer/memory.py`,
which takes `--memory` and `--precision`;
[sampling-memory.md](../internals/sampling-memory.md) breaks the peak into phases):

| profile | resident DiT GiB | peak per call GiB |
| --- | --- | --- |
| `speed` | 9.63 | 15.07 |
| `balanced` | 4.14 | 7.24 |

### Text encoder placement

`text_plan` (a `TextEncoderPlan` or its fields; `TEXT_ENCODE`, the same dict as
training's). `memory` `auto` (default) picks the fastest placement that fits next to the
DiT with `reserve_gib` free (default: the DiT forward's working set at one 1024^2 CFG
sample):

| `memory` | placement |
| --- | --- |
| `resident` | 14.8 GiB in bf16 on the device |
| `offload` | host copy, moved in per encode call and released after it |
| `ring` | one decoder layer at a time through a small ring; about 2.3 GB on the device during a call, nothing between calls (`stream` is accepted as the same value) |

- `precision="int8"` / `"fp8"` halves the weights and the per-call traffic;
  `fp8_compute` (with `fp8_recipe` row / tensor / block) and `mxfp8` run the GEMMs
  themselves in fp8 on sm89+ (8.4 / 8.8 GiB resident instead of 14.8 GiB; see
  [fp8 / MXFP8 compute](#fp8--mxfp8-compute)).
- `dtype` is `auto` (bf16, fp16 without bf16), `bf16` or `fp16`.
- `free_after_use=True` frees the encoder after its first pass (encode every prompt in
  that call).
- Prompts of a call are encoded together (`layout="packed"`, up to `batch` prompts per
  kernel call); see [text-encoder.md](../internals/text-encoder.md).
- A placement set by the user is used as given. The per-phase memory of a call is in
  [sampling-memory.md](../internals/sampling-memory.md).
- **Text device**: `text_device="cpu"` (`TEXT_DEVICE = "cpu"`) runs the text encoder on
  the CPU and moves only the embeddings to the compute device. The 9B encoder then needs
  ~18 GB of host RAM in bf16 and no VRAM; encoding speed is the CPU's. Training uses the
  CPU only for the offline cache pass.

### VAE placement

`vae_plan` (a `VaePlan` or its fields; `VAE_ENCODE`, as in training):

- `auto` (default) takes the first that fits next to the DiT of: `resident` untiled,
  `offload` (parked on the host, moved in per decode) untiled, the same two tiled, then
  a streamed tiled pass.
- `stream` and `cpu` are the other values; `tile` forces tiled passes.
- `VAE_DTYPE` sets its dtype.

### fp8 / MXFP8 compute

On sm89+, `BASE_PRECISION = "fp8_compute"` (with `FP8_RECIPE` `"row"` / `"tensor"` /
`"block"`) or `"mxfp8"` quantizes the DiT's frozen linears once at load (e4m3 weights,
1 byte each, plus their scales) and runs their GEMMs in fp8, the activations quantized
per call. `BASE_PRECISION` int8 / fp8 are the weight-only formats.

- `BASE_PRECISION = "auto"` (default) runs what the DiT file stores: the ComfyUI int8
  file its int8 ConvRot bases (resident with no profile), a 16-bit file in 16 bits;
  `"compute"` dequantizes an int8 file ([weights.md](weights.md#dit)).
- A quantized `BASE_PRECISION` places the DiT under the `speed` profile unless
  `MEMORY_PROFILE` names one (`memory_overrides={"base_precision": ..., "fp8_recipe":
  ...}` in Python).
- The text encoder takes the same formats (`TEXT_ENCODE["precision"]`, `auto` by
  default: a ComfyUI int8 / W4A8 file keeps its stored layers).

RTX 5090, 1024^2, 20 steps of 2 prompts with CFG (`scripts/probe/fp8_inference.py`):

| DiT base | weights GiB | s / step | peak GiB | velocity cos vs fp64 |
| --- | --- | --- | --- | --- |
| bf16 | 9.66 | 0.875 | 10.49 | 0.99979 |
| fp8_compute row | 5.01 | 0.567 | 6.16 | 0.99815 |
| fp8_compute tensor | 5.01 | 0.570 | 6.15 | 0.99796 |
| fp8_compute block | 5.01 | 0.638 | 6.15 | 0.99817 |
| mxfp8 | 5.33 | 0.483 | 6.48 | 0.99798 |

The reference is the host fp64 forward of the bf16 weights. A few short prompts encode
slower in fp8 than in bf16 (each GEMM is launch-bound at that size); the formats and
kernels are in [quant-kernels.md](../internals/quant-kernels.md).

### fp16

On GPUs without bf16, pass `dtype=torch.float16` (`PRECISION="fp16"`) and either:

- an `fp16_fix` (`"online"` or `"fp32_residual"`, with `fp16_fix_kwargs={"c": 0.0625}`),
  or
- an fp16-fix checkpoint as `transformer=...` (`TRANSFORMER`), which carries its fold
  (`FP16_FIX` stays `"none"`).

See [numerics.md](../concepts/numerics.md).

### Kernel tuning

`tune` (`TUNE`; default `"off"`). Inference binds the card's stored decisions, else the
default configs; `"table"` tunes a missing decision on its first use and stores it.

| `TUNE` | effect |
| --- | --- |
| `"off"` (default) | binds the decisions stored for this card in the tuning table (filled by a training run or `kohakumoss-tune`), else the default configs; times nothing. LyCORIS binds its stored tile plans the same way |
| `"table"` | also tunes a missing decision on its first call (that call is slower) and stores it, and measures the queued kernel-vs-compile problems after each generate |

See [tuning.md](../internals/tuning.md#pipelines).

### Training-time samples

`MossPipeline.from_components(backbone, text_encoder, vae)` shares the training model's
modules in place (under `no_grad`); the sample prompts are encoded once when
training starts and passed as embeddings.

## Step diagnostics

Every sampler step logs at INFO its time and, for a streamed DiT:

- the bytes the weight ring copied;
- the seconds the copy stream was busy and the seconds the compute stream waited for a
  copy;
- host synchronizations.

On an accelerator a step records one timing event, and the host queues the next step
while the device runs this one. The batch's events are read
once when it ends, its step lines are logged then, and the step time is the device time
between consecutive step events.

Each batch ends with a summary: the first step (JIT compiles and first copies land
there) and the mean of the rest, with the copy-overlap ratio (1 - wait / copy). The same
numbers are in `settings["step_stats"]` and the sample script's
`metadata-<first index>.json`.

Reading them:

- a step that is slow with little wait is compute-bound (untuned or eager ops);
- a large wait means the copies do not overlap.

The frozen ring keeps its first `stream_resident - 2` blocks on the device and cycles
the rest through two slots, so `balanced` (`stream_resident` 16, 42 blocks) keeps 14
blocks on the device and copies 28 per step (6.34 GB, 0.21 GiB per block).
RTX 3090 (Linux, fp16 fix, 1024^2, CFG packed, TUNE=table), s/step after the first:

| DiT placement | s/step | copied per step | overlap |
| --- | --- | --- | --- |
| resident | 1.557 (first step 6.92) | - | - |
| `balanced` | 1.448 | 6.34 GB in 0.49 s | no wait |
| `minimal` with `BASE_PRECISION="int8"` (int8 blocks through two slots) | 1.568 | 4.76 GB | 98% |

## Comparison grid

`scripts/sample/compare_grid.py` compares adapter checkpoints against the base model:

```bash
kogine run scripts/sample/compare_grid.py --config my_grid.py
```

```python
WEIGHTS = "ComfyUI/models"
CHECKPOINTS = [
    "runs/quality/quality-lora8/checkpoints/step_00000100/trainable.safetensors",
    "runs/quality/quality-lora8/checkpoints/step_00000200/trainable.safetensors",
    "runs/quality/quality-lora8/checkpoints/step_00000400/trainable.safetensors",
]
PROMPTS = ["a red brick house beside a lake", "a lighthouse at dusk"]
SEEDS = [0, 1]
OUT = "runs/quality/quality-lora8/grid"
```

What it renders:

- The weights are the base model, then each of `CHECKPOINTS` (a path, or
  `(path, multiplier)`; bare paths take `MULTIPLIER`), each rendered over `PROMPTS`.
- Every prompt keeps its seed (`SEEDS`, else `SEED_BASE` + index) and the sampler
  settings across weights, at its `SIZES` entry (per prompt, the last repeats, as in
  sample.py).
- The device, precision and text-encoder knobs are those of `sample.py`.

How it runs:

- The DiT loads once and the prompts are encoded once for every column (so
  `TEXT_ENCODE = {"free_after_use": True}` frees the text encoder after that pass).
- Each checkpoint is loaded onto the DiT for its pass and removed after it
  (`ADAPTER_MODE` bypass or merge, both undone exactly), so the base is the plain model.
- A training checkpoint is read in the fold space it trained in
  ([adapters.md](../concepts/adapters.md#inference)), whatever fold the DiT file
  carries.

Layout and labels:

| knob | effect |
| --- | --- |
| `GRID_LAYOUT` | `"prompt-cols"` (default): prompts on the columns, weights on the rows; `"prompt-rows"` swaps them |
| `LABELS` | weight labels; default `step N` from a training checkpoint path. Prompt labels are `seed: prompt` |
| `LABEL_LINES` | labels are word-wrapped to their cell, at most this many lines |
| `FONT_SIZE` | label font in pixels (0 = image width / 32, at least 14) |

`OUT` gets `grid.png`, every image as `r<prompt>_c<weight>.png`, and `metadata.json`
(columns, rows, files, knobs).
