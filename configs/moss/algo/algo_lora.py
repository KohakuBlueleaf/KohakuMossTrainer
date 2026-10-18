"""Algorithm layer for scripts/train/moss.py: LoRA r32 on the attention + MLP linears.
Composed after configs/moss/_base/base_lora.py; see docs/guides/configs.md."""

ALGO = "lora"
PRESET = "attn-mlp"
DIM = 32
ALPHA = 32.0
LR = 1e-4
ALGO_KWARGS = {"use_tucker": False, "use_scalar": False, "rank_dropout_scale": False}
# LyCORIS dropout / rank_dropout / module_dropout
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
