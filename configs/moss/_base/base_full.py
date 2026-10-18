"""Shared full-finetune defaults for scripts/train/moss.py: the LoRA-family base with the
full-finetune algorithm layer (configs/moss/algo/algo_full.py)."""

from kohakuengine import use_config

use_config("base_lora.py")
use_config("../algo/algo_full.py")
