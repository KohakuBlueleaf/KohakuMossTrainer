"""Uniform stream rescale of a DiT: every writer times C (None: the power of two nearest
1 / the geometric mean of the block-output RMS on the records) and every norm eps times
C^2; reports the per-block stream RMS before and after and the velocity difference with
the exact eps and with the stock eps the written file loads with.

    kogine run scripts/heal/rescale.py --set DIT=healed.safetensors \
        --set RECORDS=heal/records --set OUT=healed_scaled.safetensors
"""

import json
from pathlib import Path

import numpy as np
import torch

from kohakumoss.heal.losses import rel_sq
from kohakumoss.heal.records import load_records, noised_inputs
from kohakumoss.heal.rescale import power_of_two_scale, rescale_stream, restore_eps
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone, save_single
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.heal.rescale")

DIT = ""  # the DiT to rescale
RECORDS = ""  # heal records (scripts/heal/cache.py)
OUT = ""  # output .safetensors
C: float | None = None  # scale; None = from the records
N = 8  # records measured
SIGMA = 0.5
DEVICE = "cuda"


@torch.no_grad()
def measure(model, xs) -> tuple[list[float], list[torch.Tensor]]:
    """(geometric mean over xs of each joint block's output RMS, the velocities)."""
    acc, cur, vs = np.zeros(model.cfg.depth), {}, []

    def hook(i):
        def fn(mod, inp, out):
            cur[i] = float(out.float().square().mean().sqrt())

        return fn

    hooks = [b.register_forward_hook(hook(i)) for i, b in enumerate(model.layers)]
    for x in xs:
        with torch.autocast(x["target"].device.type, dtype=torch.bfloat16):
            vs.append(model(*x["x"]).float())
        acc += np.log([cur[i] for i in range(model.cfg.depth)])
    for h in hooks:
        h.remove()
    return np.exp(acc / len(xs)).tolist(), vs


def max_diff(ref: list[torch.Tensor], got: list[torch.Tensor]) -> float:
    return max(float(rel_sq(b, a).sqrt()) for a, b in zip(ref, got))


def main() -> None:
    setup()
    check_script(globals())
    if not (DIT and RECORDS and OUT):
        raise SystemExit("set DIT, RECORDS and OUT")
    device = torch.device(DEVICE)
    _, recs, empty = load_records(Path(RECORDS))
    model, _ = load_backbone(DIT, device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    xs = noised_inputs(recs[:N], empty, device, (SIGMA,), model.cfg.rope_axes)
    rms, v0 = measure(model, xs)
    c = float(C) if C is not None else power_of_two_scale(rms)
    before = rescale_stream(model, c)
    rms_c, v1 = measure(model, xs)
    restore_eps(model, before)
    _, v2 = measure(model, xs)
    report = {
        "c": c,
        "rms": rms,
        "rms_scaled": rms_c,
        "v_diff_exact_eps": max_diff(v0, v1),
        "v_diff_stock_eps": max_diff(v0, v2),
    }
    save_single(model, OUT, torch.bfloat16)
    Path(OUT).with_suffix(".json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    log.info(
        f"c = {c}; velocity difference exact eps {report['v_diff_exact_eps']:.2e}, stock eps {report['v_diff_stock_eps']:.2e}; wrote {OUT}"
    )


if __name__ == "__main__":
    main()
