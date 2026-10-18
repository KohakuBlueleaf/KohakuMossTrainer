# Scripts

Every script runs from the repository root, in one of three ways (the "runs as" column):

| runs as | command | settings |
| --- | --- | --- |
| kogine | `kogine run <script> --config <file> --set KEY=VALUE` | module-level knobs ([configs.md](configs.md)) |
| python | `python <script> --flag` | command-line flags |
| shell | `.\<script>.ps1` (PowerShell) or `bash <script>.sh` | script parameters |

Each script's docstring (or header) holds its full usage.

## Train, sample, export

| script | runs as | what it does |
| --- | --- | --- |
| `scripts/train/moss.py` | kogine | training: LyCORIS adapters or full finetune, every knob ([first-lora.md](first-lora.md), [training.md](../concepts/training.md)) |
| `scripts/sample/sample.py` | kogine | images from base weights and adapters, one metadata file per run ([inference.md](inference.md#cli)) |
| `scripts/sample/compare_grid.py` | kogine | comparison grid: the base model then each checkpoint over a prompt list ([inference.md](inference.md#comparison-grid)) |
| `scripts/export/export.py` | kogine | a training state's adapters as ComfyUI / LyCORIS files, or a merged / diffusers transformer ([export-to-comfy.md](export-to-comfy.md)) |
| `scripts/export/merge.py` | kogine | merge LyCORIS, ComfyUI or training-state adapter files into a base transformer |
| `scripts/export/fp16_fix.py` | kogine | the fp16-fix checkpoint: the residual fold written into the weights ([numerics.md](../concepts/numerics.md#the-fp16-fix-checkpoint)) |
| `scripts/export/fp8_quant.py` | kogine | a pre-quantized fp8 / MXFP8 transformer (`TRANSFORMER=<file>` in training and sampling) |
| `scripts/export/text_quant.py` | kogine | a pre-quantized text encoder (`stored` in `TEXT_ENCODE`) |

## Examples

| script | runs as | what it does |
| --- | --- | --- |
| `scripts/examples/train.ps1`, `train.sh` | shell | a short training run from a few options: writes the config, checks it, runs `scripts/train/moss.py` ([scripts/examples/README.md](../../scripts/examples/README.md)) |
| `scripts/examples/sample.ps1`, `sample.sh` | shell | the same for `scripts/sample/sample.py`: base weights, adapters, fp16-fix transformer, placements |
| `scripts/examples/grid.ps1`, `grid.sh` | shell | the same for `scripts/sample/compare_grid.py`: adapters, a train example run's export or checkpoints, per-artist dataset captions (`grid_prompts.py`) |

## Data

| script | runs as | what it does |
| --- | --- | --- |
| `scripts/data/inspect_data.py` | kogine | catalog and epoch-0 plan statistics, sample captions ([datasets.md](datasets.md)) |
| `scripts/data/catalog_build.py` | kogine | scan the datasets once and write their catalogs, or import them into `moss_shard` |
| `scripts/data/cache_text.py` | kogine | the offline text-embedding cache |
| `scripts/data/cache_latents.py` | kogine | the offline latent cache |
| `scripts/data/cache_merge.py` | kogine | fold an online cache's per-rank shards into the main files |
| `scripts/data/catalog_cache.py` | kogine | show, drop or compact the catalog cache files |

## Setup

| script | runs as | what it does |
| --- | --- | --- |
| `scripts/install/install.sh`, `install.ps1` | shell | torch from the index of this GPU / driver, KohakuMoss with its default extra, the doctor ([install.md](install.md)) |
| `scripts/colab/setup.py` | python | the Colab venv with the torch build of the VM's driver; `kohakumoss_colab.ipynb` is the notebook |
| `scripts/doctor/doctor.py` | python | the environment report (`kohakumoss-doctor`) |
| `scripts/bench/tune/tune.py` | kogine | fill the per-card kernel tuning table once (`kohakumoss-tune`, [logging.md](logging.md#kernel-tuning)) |

## Multi-GPU (Linux)

| script | runs as | what it does |
| --- | --- | --- |
| `scripts/dist/check_comm.py` | kogine | process-group check and collective / p2p link sweeps ([multi-gpu.md](multi-gpu.md)) |
| `scripts/dist/scaling_report.py` | kogine | the scaling table of a set of runs: samples/s, efficiency, step breakdown, loss parity |

## Benches

Each bench writes one JSON per card under `runs/bench/<area>/`; the tables in
[performance/](../performance/README.md) and [internals/](../internals/README.md) come
from them.

| script | measures |
| --- | --- |
| `scripts/bench/kernel/attention.py` | varlen attention, every backend vs the PyTorch baselines |
| `scripts/bench/kernel/norm_mod.py`, `qk_rope.py`, `residual.py`, `ffn.py`, `fused_linear.py` | the block ops vs PyTorch eager and `torch.compile` |
| `scripts/bench/kernel/quant.py` | int8 / fp8 weight-only frozen-base linears vs dequantize + matmul |
| `scripts/bench/kernel/fp8.py` | fp8 / MXFP8 compute linears vs the 16-bit `F.linear` |
| `scripts/bench/kernel/optim.py` | the optimiser step kernels vs `torch.optim.AdamW` |
| `scripts/bench/adapters/fold_base.py` | the LoRA bypass with the base add folded into the LyCORIS kernel |
| `scripts/bench/textenc/textenc.py`, `ops.py`, `quant.py` | the text encoder: tokens/s per placement and layout, its ops, its quantized formats |
| `scripts/bench/vae/vae.py` | VAE encode / decode precision, speed and memory per dtype |
| `scripts/bench/offload/stream.py`, `knobs.py`, `limits.py` | copy bandwidth, each memory knob in isolation, knob combinations against the link |
| `scripts/bench/step/step.py` | the training step per memory profile through the trainer |
| `scripts/bench/train/memory.py` | per-phase device memory of the trainer's startup and first steps |
| `scripts/bench/infer/memory.py` | per-phase device memory of one generate call |
| `scripts/bench/data/loader.py`, `catalog.py` | loader throughput; catalog scan, open and token-budget fill |
| `scripts/bench/matrix/launch.py`, `host.py` | the real-GPU bench matrix (dev package `kohakumoss_bench`, `uv pip install -e scripts/bench/matrix`): every card's speed / VRAM / quality / inference runs ([bench-matrix.md](../performance/bench-matrix.md)) |

## Probes

Real-weight checks of the model itself; each writes one JSON report.

| script | checks |
| --- | --- |
| `scripts/probe/embed.py` | encode the probe prompts once, for the other probes |
| `scripts/probe/parity.py` | model and pipeline against the official implementation |
| `scripts/probe/trajectory.py` | one sampling trajectory, official loop or ours, and their comparison |
| `scripts/probe/activations.py` | per-layer magnitudes in fp32 and the fold factor for fp16 |
| `scripts/probe/backward.py` | per-module forward / backward magnitudes in fp16 |
| `scripts/probe/adapter_e2e.py` | an adapter on the live model against its exported files loaded by `MossPipeline` |
| `scripts/probe/fp8_inference.py` | fp8 / MXFP8 inference of the text encoder and the DiT against bf16 and fp64 |
