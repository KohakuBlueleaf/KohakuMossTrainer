"""VRAM layer for scripts/train/moss.py, adapters on a 6 GB GPU: two resident streamed
blocks with activation and gradient offload and the AdamW update on the host, at 4352
tokens (a 1 MP pair: 4096 image tokens and its caption), from offline latent and text caches (the GPU holds only the backbone during
training); a checkpoint every 30 minutes as well as every SAVE_EVERY steps. Precision
stays with the precision knobs. Composed after the algorithm layer; see
docs/guides/configs.md and docs/concepts/profiles.md."""

MEMORY_PROFILE = "ckpt-ring2-actoff-optcpu-gradoff"
MAX_TOKENS = 4352
LATENT_CACHE = "offline"
TEXT_CACHE = "offline"
SAVE_MINUTES = 30
