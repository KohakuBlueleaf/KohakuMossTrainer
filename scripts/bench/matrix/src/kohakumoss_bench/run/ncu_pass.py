"""The ncu pass of one bench row: its command under `host.ncu` and the environment it
runs in. See docs/performance/bench-matrix.md#ncu."""

import json
import os
from pathlib import Path

from kohakumoss_bench.measure import ncu
from kohakumoss_bench.plan.config import Card
from kohakumoss_bench.run.proc import visible
from kohakumoss_bench.run.stage import copy_tree

PASSED = ("PATH", "CUDA_HOME", "CUDA_PATH")


def ncu_command(
    host_ncu: list[str],
    replay: str,
    env: dict,
    kernel_root: Path,
    card: Card,
    run_dir: Path,
    child: list,
) -> tuple[list, dict]:
    """(command, metadata) of the ncu pass: the child under `host_ncu`, its environment
    from an env file (sudo resets it), caches and a copy of the kernel tables inside
    the run directory (nothing the profiled process writes lands in the runner
    account's own caches)."""
    tensor, how = ncu.tensor_metric(host_ncu, card.sm)
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    kernels = run_dir / "kernel-cache"
    tuning = kernel_root / "tuning"
    if tuning.is_dir():
        copy_tree(tuning, kernels / "tuning")
    run_env = {
        **env,
        **visible(card),
        "PYTHONUNBUFFERED": "1",
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "TRITON_CACHE_DIR": str(home / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(home / "inductor"),
        "KOHAKUMOSS_KERNEL_CACHE": str(kernels),
    }
    for key in PASSED:
        if key in os.environ and key not in run_env:
            run_env[key] = os.environ[key]
    env_file = run_dir / "env.json"
    env_file.write_text(json.dumps(run_env, indent=1), encoding="utf-8")
    target = [child[0], "-u", *child[1:], "--env-file", str(env_file)]
    cmd = ncu.command(host_ncu, tensor, run_dir / ncu.REPORT, target, replay)
    meta = {"tensor_metric": tensor, "chosen_by": how, "replay": replay}
    return cmd, {**meta, "command": cmd}
