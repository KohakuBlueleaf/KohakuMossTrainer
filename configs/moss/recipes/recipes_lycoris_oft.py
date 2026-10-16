"""Diag-OFT (block 32, no constraint) on the attention + MLP linears.
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lycoris_oft.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")

RUN_NAME = "oft-b32"
ALGO = "oft"
DIM = 32
ALPHA = 0.0
ALGO_KWARGS = {"constraint": 0.0, "rescaled": False}
LR = 1e-4
