"""Heal records: rows [START, START + ROWS) of TRAIN_CONFIG's dataset (this process: the
rows r with (r - START) % PARTS == PART), each row's planned latents (VAE) and its
caption's embedding (the source template, seeded by the row), plus the empty-prompt
embedding. Run one process per GPU with PART = 0 .. PARTS - 1.

    kogine run scripts/heal/cache.py --set TRAIN_CONFIG=my_run.py --set OUT=heal/records \
        --set ROWS=17000 --set PART=0 --set PARTS=4
"""

from pathlib import Path

import torch

from kohakumoss.cli.config.validate import load_knobs
from kohakumoss.heal.records import write_records
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import resolve_config
from kohakumoss.precision import build_precision
from kohakumoss.training.build.model import build_encoders
from kohakumoss.training.knobs import check
from kohakumoss.training.step import DeviceBatcher, OnlineLatents, OnlineText
from kohakumoss_common.knobs import check_script
from kohakumoss_data import DataConfig, MossDataset

log = get_logger("scripts.heal.cache")

TRAIN_CONFIG = ""  # training config: dataset, captions, resolution, encoders
OUT = ""  # records directory
ROWS = 4096  # rows to encode
START = 0  # first row
PART = 0  # this process's part
PARTS = 1  # processes sharing the rows
DEVICE = 0  # CUDA device index
TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train" / "moss.py"


def main() -> None:
    setup()
    check_script(globals())
    if not TRAIN_CONFIG or not OUT:
        raise SystemExit("set TRAIN_CONFIG and OUT")
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", int(DEVICE))
    torch.cuda.set_device(device)
    over = {
        "TEXT_CACHE": "off",
        "LATENT_CACHE": "off",
        "SAMPLE_EVERY": 0,
        "ENCODE_DEVICES": (),
        "DEVICES": (int(DEVICE),),
        "TEXT_DEVICE": "",
        "PARALLEL": "none",
    }
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        over |= {
            "PRECISION": "fp16",
            "FP16_FIX": "online",
            "FP16_FIX_KWARGS": {"c": 0.0625},
        }
    k = check(load_knobs(TRAIN_SCRIPT) | load_knobs(Path(TRAIN_CONFIG)) | over)
    prec = build_precision(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    ds = MossDataset(DataConfig.from_knobs(k))
    vae, text, _ = build_encoders(k, device, prec)
    rope = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None).rope_axes
    batcher = DeviceBatcher(
        device,
        rope,
        text=OnlineText(text.encode),
        latents=OnlineLatents(vae.encode_images),
    )
    if PART == 0 and START == 0:
        torch.save(text.encode([""])[0].to(torch.bfloat16).cpu(), out / "empty.pt")
    rows = list(range(START + PART, START + ROWS, PARTS))
    tag = f"{PART}" if START == 0 else f"{PART}s{START}"

    def progress(n: int, total: int) -> None:
        log.info(f"part {PART}: {n}/{total} rows")

    n = write_records(ds, batcher, rows, out, tag, progress)
    log.info(f"wrote {n} records to {out}")


if __name__ == "__main__":
    main()
