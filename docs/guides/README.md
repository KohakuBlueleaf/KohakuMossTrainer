# Guides

In the order a first run follows them:

| page | content |
| --- | --- |
| [install.md](install.md) | one install command per platform (Linux, Windows, Intel Arc, Google Colab), extras, the doctor |
| [install-windows.md](install-windows.md) | Windows toolchain, feature matrix |
| [weights.md](weights.md) | where each weight file goes: the ComfyUI layout (int8 / W4A8 / bf16 files, the component knobs), the HF layout, the quantized formats |
| [first-lora.md](first-lora.md) | train a first LoRA / LyCORIS adapter, end to end |
| [configs.md](configs.md) | kogine configs, preset layers, recipes, `kohakumoss-config`, host RAM per VRAM layer |
| [datasets.md](datasets.md) | preparing data, inspecting the plan, caching, importing to the dataset format |
| [inference.md](inference.md) | sampling with base weights and adapters, comparison grids |
| [export-to-comfy.md](export-to-comfy.md) | what a run saves (SAVE_FORMAT comfy / hf / both, merged files, checkpoint exports) and exporting checkpoints for ComfyUI |
| [full-finetune.md](full-finetune.md) | full finetune, memory profiles and host RAM for it |
| [multi-gpu.md](multi-gpu.md) | DDP, FSDP2, pipeline, threaded pipeline (Linux) |
| [logging.md](logging.md) | log levels, colours, tables, progress bars, kernel tuning lines, VRAM numbers |
| [scripts.md](scripts.md) | every script under `scripts/`: what it does and how to run it, the example run scripts |
