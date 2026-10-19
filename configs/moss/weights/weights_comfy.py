"""Weights layer for scripts/train/moss.py: a ComfyUI models folder with every file
named (the defaults: the int8 ConvRot DiT and text encoder, the MOSS-Image VAE). See
docs/guides/weights.md."""

WEIGHTS = "ComfyUI/models"
TRANSFORMER = "moss_image_base_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3.5_9b_int8_convrot.safetensors"
VAE = "flux2-vae_small.safetensors"
