"""Multi-GPU sanity + link measurement: spawn one process per device itself, run
the gradient-reducer check and the all_reduce / all_gather / reduce_scatter / p2p hop
sweeps.

    kogine run scripts/dist/check_comm.py --set DEVICES=0,1 --set BACKEND=auto
Writes <OUT>/<backend>_<n>.json.
"""

import json
import os

import torch

from kohakumoss.logging.log import get_logger, setup
from kohakumoss.parallel.launch import default_backend
from kohakumoss.parallel.telemetry import selftest
from kohakumoss.types.plans import ParallelPlan
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.check_comm")

DEVICES = (0, 1)  # device indices, one process each
BACKEND = "auto"  # "auto" | "nccl" | "gloo" | "xccl"
DEVICE_TYPE = "cuda"  # "cuda" | "xpu" | "cpu"
INIT = "tcp"  # "tcp" | "file"
OUT = "runs/bench/dist"  # report directory


def run(worker, plan, device_type, *args):
    return selftest.spawn_results(worker, plan, *args, device_type=device_type)


def main():
    setup()
    check_script(globals())
    devices = tuple(int(d) for d in DEVICES)
    backend = default_backend(DEVICE_TYPE) if BACKEND == "auto" else BACKEND
    plan = ParallelPlan(kind="check", backend=backend, init=INIT, devices=devices)
    report = {"backend": backend, "devices": devices}
    report["ddp"] = run(selftest.ddp_worker, plan, DEVICE_TYPE)
    report["all_reduce"] = run(selftest.comm_worker, plan, DEVICE_TYPE)
    report["all_gather_reduce_scatter"] = run(
        selftest.collective_worker, plan, DEVICE_TYPE
    )
    report["p2p_hop"] = run(selftest.hop_worker, plan, DEVICE_TYPE)
    if torch.cuda.is_available():
        report["gpus"] = [torch.cuda.get_device_name(d) for d in devices]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{backend}_{len(devices)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    log.info(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
