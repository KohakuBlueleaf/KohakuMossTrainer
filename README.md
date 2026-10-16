# KohakuMoss

Finetuning framework for [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base):
LyCORIS algorithms (LoRA, LoHa, LoKr, DoRA, OFT, BOFT, GLoRA, IA3, DyLoRA, T-LoRA) and
full finetune on text-image data at native resolutions, plus inference.

- Own MOSS-Image implementation on packed variable-length batches, checked against the
  official code (fp32 forward relative error 4.7e-6, 50-step sampling LPIPS 2.3e-6).
- Kernels per GPU arch: TileLang for sm70 / sm75, Triton for sm80+, each op selected by
  measurement against PyTorch and `torch.compile`.
- bf16, and fp16 for GPUs without bf16 (with an fp16 overflow fix).
- Memory profiles from "speed" to "minimal": checkpointing, activation offload, weight
  streaming, gradient offload, quantized base weights.
- Multi-GPU (Linux): DDP, FSDP2, pipeline.
- Windows and Linux; outputs load in ComfyUI.

Status: under active development.

## Install

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu130   # or the CUDA build for your GPU
uv pip install -e ".[kernels,lightning]"
kohakumoss-doctor
```

Windows: see [docs/guides/install-windows.md](docs/guides/install-windows.md).

## Quickstart

```bash
# smoke run on a random tiny model, no downloads
kogine run scripts/train/moss.py --config configs/moss/smoke/smoke_tiny.py

# prepare data, then train a LoRA on a 24 GB GPU
kogine run scripts/data/catalog_build.py --config configs/data/data_kohya_style.py
kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_24g_bf16.py

# sample with the trained adapter
python scripts/sample/sample.py --help
```

Recipes for other targets live in `configs/moss/recipes/` (16 GB fp16, 12 GB streamed,
LoKr / LoHa / OFT / DoRA, full finetune with offload or FSDP2, Intel Arc).

## Docs

[docs/README.md](docs/README.md)

## License

Apache-2.0.
