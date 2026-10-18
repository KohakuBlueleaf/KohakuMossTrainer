"""Encode the probe prompts (and the empty prompt) once; later probes load the file.

kogine run scripts/probe/embed.py --set WEIGHTS=<root>/base --set OUT=runs/probe/embeds.pt
"""

from pathlib import Path

import torch

from kohakumoss.encoders import QwenTextEncoder
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.probe.official import PROBE_PROMPTS as PROMPTS
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.embed")

WEIGHTS = ""  # pipeline root or text-encoder dir
OUT = ""  # output .pt
DEVICE = "cuda"


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not OUT:
        raise SystemExit("set WEIGHTS and OUT")
    te = QwenTextEncoder.from_pretrained(WEIGHTS, DEVICE, {"dtype": "bf16"})
    embeds = [e.float().cpu() for e in te.encode(list(PROMPTS))]
    empty = te.empty().float().cpu()
    ids = te.token_ids(list(PROMPTS))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"prompts": list(PROMPTS), "embeds": embeds, "empty": empty, "ids": ids},
        OUT,
    )
    log.info(str({"lengths": [e.shape[0] for e in embeds], "empty": empty.shape[0]}))


if __name__ == "__main__":
    main()
