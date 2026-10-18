"""Algorithm layer for scripts/train/moss.py: LoKr (factor 8; w1 and w2 full, since DIM
10000 exceeds half of every w2 dimension) on the attention + MLP linears. Composed after
configs/moss/_base/base_lora.py; see docs/guides/configs.md."""

ALGO = "lokr"
DIM = 10000
ALPHA = 1.0
ALGO_KWARGS = {
    "factor": 8,
    "decompose_both": False,
    "full_matrix": False,
    "use_tucker": False,
    "rank_dropout_scale": False,
}
LR = 2e-4
# LyCORIS dropout / rank_dropout / module_dropout
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
