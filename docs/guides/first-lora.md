# Your first LoRA

1. Install (`uv pip install -e ".[kernels,dev]"`) and check the environment:
   `kohakumoss-doctor`. It lists every GPU with its arch, the installed toolchain and
   which kernel backend serves each op.
2. Smoke-test the whole loop without downloads:
   `kogine run scripts/train/moss.py --config configs/moss/smoke/smoke_tiny.py`
   (random tiny model, synthetic data, CPU or GPU, about a minute).
3. Copy a recipe, e.g. `configs/moss/recipes/recipes_lora_24g_bf16.py`, and set
   - `WEIGHTS` to the MOSS-Image directory (with `transformer/`, `text_encoder/`, `vae/`),
   - `DATASETS` to your data (`{"source": "kohya", "path": ...}` for folders of images
     with `.txt` captions),
   - `RUN_NAME`, `STEPS`, `SAMPLE_PROMPTS`.
4. Build the latent and text caches once (`scripts/data/`), then train:
   `kogine run scripts/train/moss.py --config my_lora.py`.
5. Outputs are under `runs/<RUN_NAME>/`: `metrics.jsonl`, `samples/step_X/`,
   `checkpoints/step_X/`, `export/<name>-comfy.safetensors` (load it with ComfyUI's
   LoRA loader) and `<name>-lycoris.safetensors`.

Pick the recipe by GPU:

| GPU | recipe |
| --- | --- |
| 24 GB with bf16 (3090, 4090, 50xx) | `recipes_lora_24g_bf16.py` |
| fp16-only (V100, 2080 Ti) | `recipes_lora_16g_fp16.py` |
| 12 GB | `recipes_lora_12g_stream.py` |
| Intel Arc B-series | `recipes_lora_xpu_b50.py` |
| two GPUs | `recipes_lora_ddp.py` |

Other algorithms: `recipes_lycoris_loha.py`, `recipes_lycoris_lokr.py`,
`recipes_lycoris_oft.py`, `recipes_dora.py`. Any knob can be overridden on the command
line: `--set LR=2e-4 --set DIM=16`. Resume with `--set RESUME=latest`.
