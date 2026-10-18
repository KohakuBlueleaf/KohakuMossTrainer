"""Algorithm layer for scripts/train/moss.py: Diag-OFT (block 32, no constraint) on the
attention + MLP linears. Composed after configs/moss/_base/base_lora.py; see
docs/guides/configs.md."""

ALGO = "oft"
DIM = 32
ALPHA = 0.0
ALGO_KWARGS = {"constraint": 0.0, "rescaled": False}
LR = 1e-4
