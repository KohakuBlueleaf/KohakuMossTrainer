"""VRAM layer for scripts/train/moss.py, adapters on a 12 GB GPU: H2D-only weight
streaming with four resident blocks, activation and gradient offload, host AdamW states;
the text encoder streams its layers through a ring (about 2 GB during a pass, freed
between passes), 4 batches per pass. Precision stays with the precision knobs. Composed
after the algorithm layer; see docs/guides/configs.md and docs/concepts/profiles.md."""

MEMORY_PROFILE = "ckpt-ring4-actoff-optstream-gradoff"
MAX_TOKENS = 6144
VAE_ENCODE = {"memory": "offload", "ahead": 4}
TEXT_ENCODE = {"memory": "ring", "ahead": 4, "batch": 16, "max_tokens": 8192}
