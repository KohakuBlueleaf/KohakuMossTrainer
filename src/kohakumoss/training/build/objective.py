"""The configured token loss, built once for every path that computes it (TrainStep and
the pipeline engine's last stage)."""

from kohakumoss.diffusion.objective import LOSSES
from kohakumoss_common.registry import build


def is_masked(k: dict) -> bool:
    """The loss reads a per-token mask (only `masked_mse` does)."""
    return k["LOSS"] == "masked_mse"


def build_loss(k: dict, built):
    """LOSS / LOSS_KWARGS: `(pred, target, seg, weight, mask) -> summed loss`."""
    return build(
        {"name": k["LOSS"], "masked": is_masked(k), **k["LOSS_KWARGS"]},
        LOSSES,
        token_mse=built.backbone.ops.token_mse,
    )
