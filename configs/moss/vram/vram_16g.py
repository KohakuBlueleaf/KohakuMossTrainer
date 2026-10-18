"""VRAM layer for scripts/train/moss.py, adapters on a 16 GB GPU: the `low` profile
(block checkpointing, four resident streamed blocks, activation offload, host AdamW
states with a fused step) at 8192 tokens; the text encoder streams its layers through a
ring, 4 batches per pass. Precision stays with the precision knobs. Composed after the
algorithm layer; see docs/guides/configs.md and docs/concepts/profiles.md."""

MEMORY_PROFILE = "low"
MAX_TOKENS = 8192
TEXT_ENCODE = {"memory": "stream", "ahead": 4}
