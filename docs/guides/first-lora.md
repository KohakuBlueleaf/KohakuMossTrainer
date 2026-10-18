# Your first LoRA

The steps run from the repository root of a clone, on Linux or Windows (one GPU). On
Windows use the same commands in PowerShell or `cmd`, with the venv activated:
`.venv\Scripts\Activate.ps1` in PowerShell (after `Set-ExecutionPolicy -Scope
CurrentUser RemoteSigned` once per user; the default policy refuses scripts), or
`.venv\Scripts\activate.bat` in `cmd`.

1. Install ([install.md](install.md); Windows toolchain: [install-windows.md](install-windows.md))
   and check the environment: `kohakumoss-doctor`. It lists every GPU with its arch, the
   installed toolchain, which kernel backend serves each op, and anything missing with
   the command that fixes it.
2. Download the weights (~26 GB):
   `hf download OpenMOSS-Team/MOSS-Image-Base --local-dir models/MOSS-Image-Base`.
3. Write a config for your GPU and data:
   `kohakumoss-config new train --weights models/MOSS-Image-Base --data <dataset> --name my-lora`
   (`--algo loha | lokr | oft | dora | full`; `--source folder | folder_nl | jsonl |
   parquet | webdataset | moss_shard` for other layouts; `--vram` / `--vram-tolerance` /
   `--precision` / `--gpus` to override what it reads from the GPU; see
   [configs.md](configs.md)). It writes `configs/user/my-lora.py` and
   `configs/user/my-lora_data.py`; set `STEPS`, `SAMPLE_PROMPTS` (default: the first
   caption of your dataset, or `--prompt`) and anything else in the first.
4. Check the data plan, then build the text and latent caches once (the commands are in
   the config's docstring; [datasets.md](datasets.md) explains the output):

   ```bash
   kogine run scripts/data/inspect_data.py --config configs/user/my-lora_data.py
   kogine run scripts/data/cache_text.py --config configs/user/my-lora_data.py
   kogine run scripts/data/cache_latents.py --config configs/user/my-lora_data.py
   ```

5. Train:

   ```bash
   kogine run scripts/train/moss.py --config configs/user/my-lora.py
   ```

   Any knob can be overridden on the command line (`--set LR=2e-4 --set DIM=16`);
   resume an interrupted run with `--set RESUME=latest`. The first run on a card tunes
   its kernels before step 0 and stores the decisions; later runs reuse them
   ([logging.md](logging.md#kernel-tuning)); see [What to expect](#what-to-expect).
6. Outputs are under `runs/moss/<RUN_NAME>/`: `config.json` (the knobs), `log.txt`
   (every console log line: the plan, kernel decisions, warnings, errors),
   `metrics.jsonl`, `samples/step_XXXXXXXX/`,
   `checkpoints/step_XXXXXXXX/`, and `export/<RUN_NAME>-comfy.safetensors` and
   `export/<RUN_NAME>-lycoris.safetensors`, written at the end of the run (an
   interrupted run's checkpoints export with `scripts/export/export.py`,
   [export-to-comfy.md](export-to-comfy.md)). The `comfy` file has the key layout of
   ComfyUI's MOSS-Image model (a LoRA loader node loads it there); step 7 samples it
   with this repository's pipeline.
7. Sample with the exported adapter ([inference.md](inference.md)):

   ```bash
   kohakumoss-config new sample --weights models/MOSS-Image-Base --adapter runs/moss/my-lora/export/my-lora-comfy.safetensors --name my-lora-sample
   kogine run scripts/sample/sample.py --config configs/user/my-lora-sample.py --set PROMPT="a lighthouse on a rocky coast at sunset"
   ```

For a short run from a handful of options instead of a config file, the example scripts
write and check the config for you: `scripts\examples\train.ps1` (Windows) or
`scripts/examples/train.sh` (Linux), see
[scripts/examples/README.md](../../scripts/examples/README.md).

## Recipes

Fixed recipes compose the same layers for common targets (`configs/moss/recipes/`):

| GPU | recipe |
| --- | --- |
| 24 GB with bf16 (3090, 4090, 50xx) | `recipes_lora_24g_bf16.py` |
| fp16-only (V100, 2080 Ti) | `recipes_lora_16g_fp16.py` |
| 12 GB | `recipes_lora_12g_stream.py` |
| Intel Arc B-series | `recipes_lora_xpu_b50.py` |
| two GPUs (Linux) | `recipes_lora_ddp.py` |

Other algorithms on a 24 GB bf16 card: `recipes_lycoris_loha.py`,
`recipes_lycoris_lokr.py`, `recipes_lycoris_oft.py`, `recipes_dora.py` (each composes
`vram_24g` and `precision_bf16`); for any other card or precision use
`kohakumoss-config new train --algo loha | lokr | oft | dora`. Run a recipe with
`kogine run scripts/train/moss.py --config configs/moss/recipes/<recipe>.py`, after
setting `WEIGHTS` and `DATASETS` (in a file that `use_config`s it, or with `--set`).

## Small GPUs

`kohakumoss-config new train --vram 8` (or `--vram 6`) composes the 8 GB / 6 GB VRAM
layers: weight streaming with two resident blocks, activation and gradient offload,
host AdamW states, 4352-token batches and a checkpoint every 30 minutes
(`SAVE_MINUTES`) besides every `SAVE_EVERY` steps; the 6 GB layer needs the offline
latent and text caches of step 4 ([configs.md](configs.md#layers)). These layers keep
the DiT in host memory: see [host RAM](configs.md#host-ram) before choosing one.

At 4352 tokens a 1 MP pair fits: 4096 image tokens and a caption of up to 256 tokens.
A longer caption is over the budget, so the base's `OVERLONG = "drop_tokens"` keeps
`4352 - text tokens` of that sample's image tokens (a seeded draw,
[data.md](../concepts/data.md#resolution-batching-over-long-samples)). To keep whole
images with long captions, lower the area: `--set TARGET_AREA=802816` (896^2) or
`--set TARGET_AREA=589824` (768^2), in the train config before the latent cache pass
(the target size is part of the latent key).

## SFT weights

To train on MOSS-Image-SFT, download it
(`hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir models/MOSS-Image-SFT`) and set
the SFT arch preset and weights: `--set VARIANT=sft --set WEIGHTS=models/MOSS-Image-SFT`.
When that folder holds only the transformer, point the encoders at the base download:
`--set TEXT_ENCODER=models/MOSS-Image-Base/text_encoder --set
VAE=models/MOSS-Image-Base/vae` (and the same paths in the data config's
`TEXT_ENCODER_KWARGS` / `IMAGE_ENCODER_KWARGS`).

## What to expect

- **First run on a card**: the kernel prewarm before step 0 times every kernel problem
  the run plans and stores the decisions; its log line is `prewarm: N problems from the
  tuning table, M tuned in X s`. Later runs on the same card with the same shapes read
  the table and time nothing (`M` = 0). `kohakumoss-tune` fills the table ahead of a run
  ([logging.md](logging.md#kernel-tuning)); `--set TUNE=off` skips all timing (stored
  decisions or fixed defaults) for a quick smoke run.
- **Speed and VRAM per card**: the training bar shows s/it and `vram` (peak allocated /
  peak reserved / device use, [logging.md](logging.md#vram)). The per-card step times
  and nvidia-smi peaks of every VRAM layer come from the real-GPU bench matrix's speed
  and cap runs, in its `report.md` ([bench-matrix.md](../performance/bench-matrix.md#report)); the
  docs carry no per-layer training table.
- **Host RAM**: [configs.md](configs.md#host-ram).

## fp8 compute

On sm89+ (4090, 50xx) the frozen base and the text encoder can run their GEMMs in fp8:
`--set BASE_PRECISION=fp8_compute --set FP8_DGRAD=16bit` (`mxfp8` on Blackwell;
`FP8_RECIPE` picks the scale granularity of `fp8_compute`) and `"precision":
"fp8_compute"` in `TEXT_ENCODE`. The base then holds 1 byte per weight
(`FP8_DGRAD="fp8"`: 2, with an fp8 input gradient). The precision knobs apply under any
memory profile ([profiles.md](../concepts/profiles.md#precision-knobs)); see
[quant-kernels.md](../internals/quant-kernels.md#fp8-compute) for the formats and the
measured speed.
