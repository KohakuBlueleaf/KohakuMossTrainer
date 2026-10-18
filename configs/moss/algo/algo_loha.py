"""Algorithm layer for scripts/train/moss.py: LoHa r16 on the attention + MLP linears.
Composed after configs/moss/_base/base_lora.py; see docs/guides/configs.md."""

ALGO = "loha"
DIM = 16
ALPHA = 8.0
LR = 2e-4
ALGO_KWARGS = {"use_tucker": False, "use_scalar": False, "rank_dropout_scale": False}
# LyCORIS dropout / rank_dropout / module_dropout
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
