"""`compile_setup.apply()` gives the process a compile worker pool that is never
quiesced, in every thread, whether inductor's config was loaded before or after it: a
pool built in a new thread the way inductor builds it has no idle Timer and runs its
jobs. Without `apply()` the pool has the Timer. Each case runs in a fresh process (the
setting is process-wide)."""

import json
import os
import subprocess
import sys

import pytest

from kohakumoss_kernels.compile_setup import QUIESCE_ENV
from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="inductor compiles in-process on Windows (compile_threads=1): no pool",
    ),
]
CHILD = """
import json, sys, threading, time
from concurrent.futures import wait
from functools import partial
order, setup = sys.argv[1], sys.argv[2] == "1"
from kohakumoss_kernels import compile_setup
if order == "loaded":
    import torch._inductor.config
if setup:
    compile_setup.apply()
import torch._inductor.config as config
from torch._inductor.compile_worker.subproc_pool import SubprocPool
out = {}
def build():
    out["quiesce"] = config.quiesce_async_compile_pool
    out["pool"] = SubprocPool(2, quiesce=config.quiesce_async_compile_pool)
thread = threading.Thread(target=build)
thread.start()
thread.join()
pool = out.pop("pool")
out["timer"] = pool.timer is not None
if not out["timer"]:
    futures = [pool.submit(partial(time.sleep, 0)) for _ in range(50)]
    done, pending = wait(futures, timeout=30)
    out.update(done=len(done), pending=len(pending))
pool.shutdown()
print("RESULT " + json.dumps(out), flush=True)
"""


def pool_state(order: str, setup: bool) -> dict:
    """The child's pool: its quiesce setting (read in a new thread), Timer, jobs done."""
    env = {k: v for k, v in os.environ.items() if k != QUIESCE_ENV}
    run = subprocess.run(
        [sys.executable, "-c", CHILD, order, "1" if setup else "0"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, run.stdout[-2000:] + run.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT ") :])


def never_quiesced(state: dict) -> None:
    assert state["quiesce"] is False and state["timer"] is False, state
    assert state.get("done") == 50 and state.get("pending") == 0, state


@pytest.mark.parametrize("order", ["loaded", "fresh"])
def test_the_compile_pool_is_never_quiesced(order):
    never_quiesced(pool_state(order, setup=True))


def test_without_the_setting_the_pool_quiesces():
    default = pool_state("loaded", setup=False)
    assert default["quiesce"] is True and default["timer"] is True, default
    must_fail(lambda: never_quiesced(default), "inductor's default quiescing pool")
