"""Merge a heal run's parts (RUN/final, or a RUN/step_<n> snapshot at anneal t = 1) into
one plain single-file DiT.

    kogine run scripts/heal/merge.py --set RUN=heal/runs/heal5k --set OUT=healed.safetensors
"""

import json
from pathlib import Path

from kohakumoss.heal.config import HealConfig
from kohakumoss.heal.parts import merge
from kohakumoss.logging.log import get_logger, setup
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.heal.merge")

RUN = ""  # heal run directory (holds heal.json)
PARTS = "final"  # parts directory inside RUN
OUT = ""  # output .safetensors


def main() -> None:
    setup()
    check_script(globals())
    if not RUN or not OUT:
        raise SystemExit("set RUN and OUT")
    c = HealConfig(**json.loads((Path(RUN) / "heal.json").read_text(encoding="utf-8")))
    log.info(json.dumps(merge(c, Path(RUN) / PARTS, Path(OUT))))


if __name__ == "__main__":
    main()
