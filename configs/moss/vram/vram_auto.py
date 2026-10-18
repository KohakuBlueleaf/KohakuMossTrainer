"""VRAM layer for scripts/train/moss.py, any GPU: the memory planner takes the fastest
profile whose estimate fits the device, and the text encoder and VAE the fastest
placement that fits next to that profile's step. `kohakumoss-config new` composes it
above the largest sized layer (24 GiB). Composed after the algorithm layer; see
docs/guides/configs.md."""

MEMORY_PROFILE = "auto"
TEXT_ENCODE = {"memory": "auto", "ahead": 4}
VAE_ENCODE = {"memory": "auto"}
