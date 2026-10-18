"""LoHa r16 on the attention + MLP linears, with the 24 GB bf16 VRAM and precision
layers. Other cards: `kohakumoss-config new train --algo loha`. Drives
scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lycoris_loha.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../algo/algo_loha.py")
use_config("../vram/vram_24g.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "loha-r16"
