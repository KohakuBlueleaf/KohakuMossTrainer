"""LoHa r16 on the attention + MLP linears. Drives scripts/train/moss.py.

kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lycoris_loha.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")

RUN_NAME = "loha-r16"
ALGO = "loha"
DIM = 16
ALPHA = 8.0
LR = 2e-4
