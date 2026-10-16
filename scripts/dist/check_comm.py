"""Multi-GPU sanity + bandwidth: spawn one process per device (no torchrun), run the
DDP / pipeline / FSDP2 self-checks and an all_reduce bandwidth sweep.

    python scripts/dist/check_comm.py --devices 0,1 --backend auto
Writes runs/bench/dist/<backend>_<n>.json.
"""

import argparse
import json
import os
import tempfile

import torch

from kohakumoss.parallel import selftest
from kohakumoss.parallel.launch import default_backend, spawn
from kohakumoss.types.plans import ParallelPlan


def run(worker, plan, device_type, *args):
    path = os.path.join(tempfile.mkdtemp(), "res")
    spawn(worker, plan, path, *args, device_type=device_type)
    return selftest.read_results(path, len(plan.devices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--device-type", default="cuda")
    ap.add_argument("--init", default="tcp")
    ap.add_argument("--out", default="runs/bench/dist")
    args = ap.parse_args()
    devices = tuple(int(d) for d in args.devices.split(","))
    backend = default_backend(args.device_type) if args.backend == "auto" else args.backend
    plan = ParallelPlan(kind="check", backend=backend, init=args.init, devices=devices)
    report = {"backend": backend, "devices": devices}
    report["ddp"] = run(selftest.ddp_worker, plan, args.device_type)
    report["pipeline_1f1b"] = run(selftest.pipeline_worker, plan, args.device_type, "1f1b")
    if args.device_type == "cuda" and backend == "nccl":
        report["fsdp2"] = run(selftest.fsdp_worker, plan, args.device_type)
    report["all_reduce"] = run(selftest.comm_worker, plan, args.device_type)
    if torch.cuda.is_available():
        report["gpus"] = [torch.cuda.get_device_name(d) for d in devices]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{backend}_{len(devices)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
