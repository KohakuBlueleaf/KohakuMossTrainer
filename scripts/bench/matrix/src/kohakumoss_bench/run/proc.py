"""Process helpers of the host runner: the environment exposing one card, a run as the
leader of its own process group, ending and reaping that whole group (through sudo for
one sudo started), and the result checks of a finished run.
See docs/performance/bench-matrix.md#failures.
"""

import os
import signal
import subprocess
import sys

from kohakumoss_bench.plan.config import Card

WINDOWS = sys.platform == "win32"


def visible(card: Card) -> dict:
    """Environment exposing only `card` (PCI bus order on CUDA)."""
    if card.device_type == "xpu":
        return {"ZE_AFFINITY_MASK": str(card.index)}
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(card.index)}


def session() -> dict:
    """Popen keywords that start a run as the leader of its own process group (a new
    POSIX session; a new process group on Windows), so `end` and `reap` reach every
    process it starts (loader workers, their forkserver, a profiler's target)."""
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def signal_group(
    proc: subprocess.Popen, sig: str, sudo: list[str], leader: bool = True
) -> None:
    """Send SIG`sig` ("TERM" | "KILL") to the process group `proc` leads (`session`);
    through `sudo` (e.g. ["sudo", "-n"]) to the group and, with `leader` (`proc` not
    yet waited for, so its pid is still its own), to `proc` itself (sudo relays it to a
    command it runs in a group of its own). Windows: the whole process tree, forced
    (`taskkill /T /F`)."""
    if WINDOWS:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        return
    if sudo[:1] == ["sudo"]:
        targets = (f"-{proc.pid}", str(proc.pid)) if leader else (f"-{proc.pid}",)
        for target in targets:
            subprocess.run(
                [*sudo, "kill", f"-{sig}", "--", target],
                capture_output=True,
                check=False,
            )
        return
    try:
        os.killpg(proc.pid, getattr(signal, f"SIG{sig}"))
    except (ProcessLookupError, PermissionError):
        pass


def end(proc: subprocess.Popen, sudo: list[str]) -> None:
    """SIGTERM to the run's process group, then SIGKILL after a minute
    (`signal_group`)."""
    for sig in ("TERM", "KILL"):
        signal_group(proc, sig, sudo)
        try:
            proc.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:
            continue
    proc.wait()


def reap(proc: subprocess.Popen, sudo: list[str]) -> None:
    """SIGKILL whatever is left of a finished run's process group (a worker or a
    forkserver that outlived its trainer); nothing on Windows, whose loader workers
    exit with their parent."""
    if not WINDOWS:
        signal_group(proc, "KILL", sudo, leader=False)


def same_card(table_dir: str, name: str) -> bool:
    """Whether a tuning directory (named after the card) belongs to `name`."""
    norm = "".join(ch for ch in name.lower() if ch.isalnum())
    return bool(norm) and norm in "".join(
        ch for ch in table_dir.lower() if ch.isalnum()
    )


def nonfinite(summary: dict) -> bool:
    return bool(summary.get("nonfinite_steps")) or bool(summary.get("loss_nan_rows"))


def retuned(summary: dict) -> bool:
    """Whether the kernel or the adapter prewarm of a run timed anything."""
    return summary.get("prewarm_tuned", 0) > 0 or summary.get("adapter_measured", 0) > 0
