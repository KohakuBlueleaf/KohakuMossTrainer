"""Tune the MOSS ops once for one card / dtype / size list and store the decisions in the
per-card tuning table (what TUNE="table" binds later); same as `kohakumoss-tune`.

    kogine run scripts/bench/tune/tune.py --set PRECISION=bf16 --set SIZES=1024x1024,832x1216
    kogine run scripts/bench/tune/tune.py --set PRECISION=fp16 --set BACKWARD=true
"""

from kohakumoss.cli.tune import tune
from kohakumoss.logging.log import setup
from kohakumoss_common.knobs import check_script

_RENAMED = {"DTYPE": "renamed to PRECISION (the compute dtype, as in training)"}

DEVICE = "cuda"  # device whose card is tuned
PRECISION = "bf16"  # compute dtype, as training's PRECISION: "bf16" | "fp16"
SIZES: list = ["1024x1024"]  # "HxW" or (H, W) image sizes to cover
VARIANT = "base"  # architecture preset
TEXT_TOKENS = 512  # longest prompt in tokens
BATCH = 4  # images per generate batch (inference) / samples per batch (training)
BACKWARD = False  # True: training shapes, forward + backward
TRAINABLE = (
    "adapter"  # with BACKWARD: "adapter" (frozen base) | "full" (every weight trains)
)
KERNELS = {}  # op -> backend pins


def _size(value) -> tuple[int, int]:
    if isinstance(value, str):
        h, w = value.lower().split("x")
        return int(h), int(w)
    return int(value[0]), int(value[1])


def main() -> None:
    setup()
    check_script(globals(), _RENAMED)
    sizes = [_size(s) for s in SIZES]
    tune(
        DEVICE,
        PRECISION,
        sizes,
        VARIANT,
        TEXT_TOKENS,
        BATCH,
        BACKWARD,
        KERNELS,
        TRAINABLE,
    )


if __name__ == "__main__":
    main()
