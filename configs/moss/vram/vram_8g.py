"""VRAM layer for scripts/train/moss.py, adapters on an 8 GB GPU: the `minimal` profile
(two resident streamed blocks, activation and gradient offload, host AdamW states) at
4352 tokens (a 1 MP pair: 4096 image tokens and its caption); the text encoder streams its layers through a ring (nothing held between
passes), 8 batches per pass; a checkpoint every 30 minutes as well as every SAVE_EVERY
steps. Precision stays with the precision knobs. Composed after the algorithm layer; see
docs/guides/configs.md and docs/concepts/profiles.md."""

MEMORY_PROFILE = "minimal"
MAX_TOKENS = 4352
VAE_ENCODE = {"memory": "offload", "ahead": 8, "tile": True}
TEXT_ENCODE = {"memory": "stream", "ahead": 8, "batch": 16, "max_tokens": 8192}
SAVE_MINUTES = 30
