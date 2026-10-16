# KohakuMoss

Finetuning framework for [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base):
LyCORIS algorithms (LoRA, LoHa, LoKr, DoRA, OFT, BOFT, GLoRA, IA3, ...) and full finetune
on text-image data at native resolutions.

- Own MOSS-Image implementation with packed variable-length batches.
- Kernels per GPU arch: TileLang for sm70 / sm75, Triton for sm86 / sm89 / sm120.
- bf16 and fp16 training (fp16 for GPUs without bf16).
- Speed / VRAM profiles, activation and weight offload, DDP / FSDP2 / pipeline.
- Windows and Linux.
- Outputs load in ComfyUI.

Status: under construction.

## Install

```bash
uv pip install -e ".[kernels,dev]"
```

## License

Apache-2.0.
