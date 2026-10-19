"""Offline text-embedding cache pass: every caption variant of every row + empty prompt.

    kogine run scripts/data/cache_text.py --config configs/data/kohya/kohya_style.py \
        --set TEXT_ENCODER_KWARGS='{"path": "<weights>/base/text_encoder"}'

TEXT_ENCODER is a dotted factory returning an EncodeText callable; the default,
`kohakumoss.encoders.text_encoder`, loads the real Qwen3.5 weights at
TEXT_ENCODER_KWARGS["path"] on the GPU, placed by `plan` (TextEncoderPlan fields; an
unset memory is "auto", the fastest placement that fits; e.g. {"memory": "cpu"} for a
host-only pass), with `device` and `dtype`; TEXT_ENCODE's dtype / precision /
quant_group / fp8_recipe (the train config's, through use_config) fill the plan
fields the kwargs leave unset, so the pass computes what the run keys on. The encoder's `encoder_id` must equal
TEXT_ENCODER_ID (a stand-in encoder cannot write under the real id); its compute spec
(`encoder_spec`: dtype / precision) enters the cache keys as TEXT_ENCODER_SPEC, so a
run computing differently does not read these entries. ENCODE_BATCH captions go through
one encoder pass. Captions are rendered with the same CAPTION template,
CAPTION_VARIANTS and SEED as training.
"""

import logging
import sys

from tqdm.auto import tqdm

from kohakumoss.encoders.specs import text_encoder_kwargs
from kohakumoss_common.knobs import check_script
from kohakumoss_common.log import get_logger, setup
from kohakumoss_data.cache.build import build_text
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import build_encoder, data_config

log = get_logger("scripts.cache_text")

# ===================== DATA =====================
DATASETS = []
CAPTION = {"template": "tags", "keep": ["artist"], "shuffle": True, "tag_dropout": 0.0}
CAPTION_VARIANTS = 4
CACHE_DIR = ""
CACHE_DTYPE = "bf16"  # "bf16" | "fp16" | "fp32" | "fp8"
TEXT_ENCODER_ID = "qwen3.5-9b"
SEED = 0

# ===================== ENCODER =====================
TEXT_ENCODER = "kohakumoss.encoders.text_encoder"
TEXT_ENCODER_KWARGS = {"path": "", "device": "cuda", "plan": {}}
TEXT_ENCODE = (
    {}
)  # the train config's TEXT_ENCODE: its dtype / precision fields fill the plan
ENCODE_BATCH = 512


def main() -> None:
    setup()
    check_script(globals(), owners=("../train/moss.py", "."))
    encode = build_encoder(TEXT_ENCODER, text_encoder_kwargs(globals()))
    spec = getattr(encode, "encoder_spec", "")
    cfg = data_config(
        globals(), latent_cache="off", text_cache="offline", text_encoder_spec=spec
    )
    dataset = MossDataset(cfg)
    quiet = not logging.getLogger("kohakumoss").isEnabledFor(logging.INFO)
    bar = tqdm(unit="caption", file=sys.stderr, disable=quiet)

    def progress(n: int) -> None:
        bar.update(n - bar.n)

    n = build_text(dataset, encode, ENCODE_BATCH, progress)
    bar.close()
    log.info(f"wrote {n} embeddings to {cfg.cache_dir}")


if __name__ == "__main__":
    main()
