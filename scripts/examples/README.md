# Example run scripts

Each example writes a complete config for its script from a few options, checks it with
`python -m kohakumoss.cli.config.main check`, then runs it. The PowerShell file
(Windows) and the bash file (Linux) take the same options: `-Algo lokr` in PowerShell is
`--algo lokr` in bash. Run them from a clone, with the KohakuMoss venv activated
(`kogine`, `kohakumoss-tune` and `python` on `PATH`).

| script | drives | what it does |
| --- | --- | --- |
| `train.ps1` / `train.sh` | `scripts/train/moss.py` | a short LoRA / LoKr run on your dataset: algorithm, precision, base precision, memory profile (or the manual memory knobs), caches, sampling, tuning |
| `sample.ps1` / `sample.sh` | `scripts/sample/sample.py` | images from the base weights and adapters: precision, fp16 fix or fp16-fix transformer, DiT memory profile and base precision, text-encoder and VAE placement; `-Compare` adds a bf16 render of the stock transformer next to a `-Transformer` |
| `grid.ps1` / `grid.sh` | `scripts/sample/compare_grid.py` | the base model then each adapter (files, a train example run's export, or its checkpoints) at each multiplier, over your prompts or captions read from a kohya-style dataset per artist tag (`grid_prompts.py`, a kogine script the grid runs with a generated `prompts_config.py`) |

Paths:

- weights: `-Weights` / `--weights`, else `KOHAKUMOSS_WEIGHTS`: a ComfyUI models folder
  (`diffusion_models/`, `text_encoders/`, `vae/`, e.g. `ComfyUI/models` from the
  README's download; the int8 files are the defaults), or an HF pipeline folder with
  `transformer/`, `text_encoder/`, `vae/` (`models/MOSS-Image-Base`);
- dataset: `-Dataset` / `--dataset`, else `KOHAKUMOSS_DATASET` (kohya-style folders, or
  `-Source folder` for image + `.txt` pairs in the train example);
- output: `-Out` / `--out`, default `runs/examples/<script>` under the current directory;
  each run gets `<Out>/<yyyyMMdd-HHmmss>-<device>-...` with its generated `config.py`
  and the script's outputs. The grid's `-Run <name>` reads the train example's runs
  under `-TrainOut` (default `runs/examples/train`).

Device: `-Device cuda` (NVIDIA, `-Index` sets `CUDA_VISIBLE_DEVICES` in nvidia-smi order)
or `-Device xpu` (Intel Arc, `-Index` sets `ZE_AFFINITY_MASK`). The PowerShell scripts
restore those variables when they exit; the bash scripts set them for the child
processes only.

Common flags: `-DryRun` / `--dry-run` writes and checks the config without running;
`-TuneOnly` / `--tune-only` (train, sample) runs `kohakumoss-tune` on the run's shapes
(no weights loaded), so the first real run binds stored kernel decisions; `-Cache off |
online | offline` (train; default `off`: the VAE and text encoder encode every batch).
Each `.ps1` parameter has its meaning in the comment above it; `--help` on a `.sh` lists
the flags with their defaults.

```powershell
.\scripts\examples\train.ps1 -Weights ComfyUI\models -Dataset D:\data\my_set -Steps 30 -SampleEvery 10
.\scripts\examples\sample.ps1 -Weights ComfyUI\models -Adapter runs\examples\train\<run>\export\<run>-lycoris.safetensors
.\scripts\examples\grid.ps1 -Weights ComfyUI\models -Run <run> -Checkpoints -Prompt "a lighthouse at dusk"
```

```bash
bash scripts/examples/train.sh --weights ComfyUI/models --dataset /data/my_set --steps 30 --sample-every 10
bash scripts/examples/sample.sh --weights ComfyUI/models --adapter runs/examples/train/<run>/export/<run>-lycoris.safetensors
bash scripts/examples/grid.sh --weights ComfyUI/models --run <run> --checkpoints --prompt "a lighthouse at dusk"
```

Memory: every example defaults to `-Memory auto`, the fastest memory profile whose
estimate fits the card's free memory (in sampling, one 1024x1024 CFG sample), so the
same command runs on an 8 GB and a 32 GB card. `-Memory minimal` (two resident DiT
blocks, the rest streamed from host RAM) or any catalog name fixes the placement,
`-Memory ""` keeps the whole DiT resident in sampling (about 10 GB in 16 bit), and
`-BasePrecision int8` halves the streamed weights. The configs follow
[docs/guides/configs.md](../../docs/guides/configs.md), the memory profiles
[docs/concepts/profiles.md](../../docs/concepts/profiles.md), the sampling knobs
[docs/guides/inference.md](../../docs/guides/inference.md).
