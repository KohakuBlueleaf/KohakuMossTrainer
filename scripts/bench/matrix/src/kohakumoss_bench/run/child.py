"""One bench run in its own process: optional VRAM cap, the script through
KohakuEngine with its config, and on an exception a `failure.json` that classifies it
(the host runner decides from it whether the host's queue goes on).

    python -m kohakumoss_bench.run.child --script scripts/train/moss.py \
        --config <run>/config.py --failure <run>/failure.json [--cap-gib 8 --allow-gib 1.5]
        [--env-file <run>/env.json]

`--env-file` sets the variables it holds before any device is touched (a run started
through `sudo ncu`, whose environment sudo resets; the profiler's own variables stay).

Classes: "oom" (the setting does not fit the card), "config" (the knobs were rejected
before any work: a validation SystemExit / ValueError, or a missing optional package),
"terminated" (SIGTERM: `kohakumoss.training.shutdown` turns it into SystemExit(143)),
"crash" (anything else: an exception or a failed assertion inside a module, which stops
the host's queue for review). See docs/performance/bench-matrix.md#failures.
"""

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path

import torch
from kohakuengine import run

from kohakumoss.training import shutdown

TERMINATED = 128 + signal.SIGTERM
CONFIG_MARKERS = (
    "invalid configuration",
    "invalid memory plan",
    "package installed",
    "export: ",
)


def classify(err: BaseException) -> str:
    """ "oom" | "config" | "terminated" | "crash" of an exception that ended a run."""
    if isinstance(err, torch.OutOfMemoryError):
        return "oom"
    if isinstance(err, SystemExit) and err.code == TERMINATED:
        return "terminated"
    text = str(err).lower()
    if isinstance(err, (SystemExit, ValueError)) and any(
        m in text for m in CONFIG_MARKERS
    ):
        return "config"
    return "crash"


def frames(err: BaseException) -> list[dict]:
    """The traceback frames (file, line, function), innermost last."""
    return [
        {"file": f.filename, "line": f.lineno, "function": f.name}
        for f in traceback.extract_tb(err.__traceback__)
    ]


def cap(device_type: str, cap_gib: float, allow_gib: float) -> dict:
    """Cap the caching allocator at `cap_gib - allow_gib` on visible card 0."""
    mod = torch.get_device_module(device_type)
    total = mod.get_device_properties(0).total_memory
    limit = max(cap_gib - allow_gib, 0.5) * 1024**3
    fraction = min(limit / total, 1.0)
    mod.set_per_process_memory_fraction(fraction, 0)
    return {"cap_gib": cap_gib, "allocator_limit_gib": limit / 1024**3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--failure", required=True)
    ap.add_argument("--cap-gib", type=float, default=0.0)
    ap.add_argument("--allow-gib", type=float, default=1.5)
    ap.add_argument("--device-type", default="cuda")
    ap.add_argument("--env-file", default="")
    args = ap.parse_args(argv)
    if args.env_file:
        os.environ.update(json.loads(Path(args.env_file).read_text(encoding="utf-8")))
    shutdown.install()
    try:
        if args.cap_gib > 0:
            info = cap(args.device_type, args.cap_gib, args.allow_gib)
            print(f"bench cap: {json.dumps(info)}", flush=True)
        run(args.script, config_path=args.config)
    except BaseException as err:  # noqa: BLE001
        if isinstance(err, SystemExit) and err.code in (0, None):
            return 0
        record = {
            "class": classify(err),
            "type": type(err).__qualname__,
            "message": str(err)[:4000],
            "traceback": traceback.format_exc()[-20000:],
            "frames": frames(err),
        }
        Path(args.failure).write_text(json.dumps(record, indent=1), encoding="utf-8")
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
