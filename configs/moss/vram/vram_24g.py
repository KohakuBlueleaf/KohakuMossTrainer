"""VRAM layer for scripts/train/moss.py, adapters on a 24 GB GPU: the `balanced` profile
(block checkpointing, 16 resident streamed blocks) at 12288 tokens; the text encoder
streams its layers through a ring, 8 batches per pass. Precision stays with the
precision knobs. Composed after the algorithm layer; see docs/guides/configs.md and
docs/concepts/profiles.md."""

MEMORY_PROFILE = "balanced"
MAX_TOKENS = 12288
TEXT_ENCODE = {"memory": "ring", "ahead": 8}
