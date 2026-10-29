"""Massive-channel calibration report of a DiT on heal records: the 32 largest channels,
per-site stream stats, and per tau the shrink scales D, each norm's divisor ratios and
the training-free shrink's stream stats and velocity error (CALIB records calibrate,
HELD others measure, at SIGMAS with fixed noise).

    kogine run scripts/heal/calibrate.py --set DIT=moss_image_base_bf16.safetensors \
        --set RECORDS=heal/records --set OUT=heal/calibration.json
"""

import json
from pathlib import Path

import torch

from kohakumoss.heal.calibrate import calibrate
from kohakumoss.heal.records import load_records, noised_inputs
from kohakumoss.heal.sites import tapped
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.opset import OpSet
from kohakumoss.precision import autocast_factory, build_precision
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.heal.calibrate")

DIT = ""  # the original DiT (single file / dir)
RECORDS = ""  # heal records (scripts/heal/cache.py)
OUT = ""  # report .json
TAUS = (16.0, 8.0, 4.0)  # shrink every channel above tau x the median channel
CALIB = 16  # calibration records
HELD = 16  # held-out records (after the calibration ones)
SIGMAS = (0.1, 0.3, 0.5, 0.7, 0.9)  # noise levels, cycled over the records
DEVICE = "cuda"


def main() -> None:
    setup()
    check_script(globals())
    if not (DIT and RECORDS and OUT):
        raise SystemExit("set DIT, RECORDS and OUT")
    device = torch.device(DEVICE)
    _, recs, empty = load_records(Path(RECORDS))
    calib, held = recs[:CALIB], recs[CALIB : CALIB + HELD]
    model, _ = load_backbone(DIT, device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    prec = build_precision("bf16", (), "none", {})
    taps = tapped(model, OpSet.resolve(device, prec.compute, {}))
    rope = model.cfg.rope_axes
    xs = {}
    for name, part in (("calib", calib), ("held", held)):
        per = [
            noised_inputs(
                [r], empty, device, (SIGMAS[i % len(SIGMAS)],), rope, seed=r["row"]
            )
            for i, r in enumerate(part)
        ]
        xs[name] = [x for group in per for x in group]
    report = calibrate(
        model,
        taps,
        xs["calib"],
        xs["held"],
        [float(t) for t in TAUS],
        autocast_factory(prec, device),
    )
    report["dit"] = Path(DIT).name
    Path(OUT).write_text(json.dumps(report), encoding="utf-8")
    for tau, rec in report["taus"].items():
        errs = sorted(rec["v_rel_err"])
        log.info(
            f"tau {tau}: {len(rec['shrunk'])} channels shrunk, median velocity error {errs[len(errs) // 2]:.4f}"
        )
    log.info(f"massive channels: {report['massive'][:8]}; wrote {OUT}")


if __name__ == "__main__":
    main()
