"""Log catalog and plan statistics for a data config, plus sample captions.

kogine run scripts/data/inspect_data.py --config configs/data/kohya/kohya_style.py
"""

import json

from kohakumoss_common.knobs import check_script
from kohakumoss_common.log import get_logger, setup
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import data_config
from kohakumoss_data.stats import catalog_stats, plan_stats

log = get_logger("scripts.inspect_data")

# ===================== DATA =====================
DATASETS = []
RESOLUTION_MODE = "resize"  # "resize" | "bucket" | "shifted_square" | "square_crop"
TARGET_AREA = 1024 * 1024
SIZE_STEP = 16
ASPECT_RANGE = (0.25, 4.0)
UPSCALE = "never"
BATCHING = "tokens"  # "tokens" | "tokens_ffd" | "count"
MAX_TOKENS = 16384
OVERLONG = "drop"  # "drop" | "truncate_text" | "drop_tokens"
BATCH_SIZE = 4
CAPTION = {"template": "tags", "keep": ["artist"], "shuffle": True, "tag_dropout": 0.0}
CAPTION_VARIANTS = 4
SEED = 0
SHOW_CAPTIONS = 3


def main() -> None:
    setup()
    check_script(globals(), owners=("../train/moss.py", "."))
    cfg = data_config(globals(), latent_cache="off", text_cache="off")
    dataset = MossDataset(cfg)
    budget = None if cfg.batching == "count" else cfg.max_tokens
    report = {
        "catalog": catalog_stats(dataset.catalog),
        "plan": plan_stats(dataset, 0, budget),
        "captions": [
            dataset.caption(i, 0)
            for i in range(min(SHOW_CAPTIONS, len(dataset.catalog)))
        ],
    }
    log.info(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
