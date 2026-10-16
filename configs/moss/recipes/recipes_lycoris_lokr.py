"""LoKr (factor 8, full w1, decomposed w2) on the attention + MLP linears.
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lycoris_lokr.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")

RUN_NAME = "lokr-f8"
ALGO = "lokr"
DIM = 10000
ALPHA = 1.0
ALGO_KWARGS = {"factor": 8, "decompose_both": False, "full_matrix": False}
LR = 2e-4
