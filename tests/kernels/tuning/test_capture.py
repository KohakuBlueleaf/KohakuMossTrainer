"""The tuner's CUDA-graph timing captures thread-locally: another thread synchronising
its own device-to-host copies meanwhile (an online cache writer) keeps working. Each
mode runs in a child process, since a broken capture can leave the context unusable."""

import subprocess
import sys

import pytest
import torch

from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda

CHILD = """
import sys, threading, torch
from kohakumoss_kernels.tuning import timing

mode = sys.argv[1]
errors, stop = [], threading.Event()


def writer():
    x = torch.randn(1 << 16, device="cuda")
    while not stop.is_set():
        try:
            x.cpu()
        except Exception as err:
            errors.append(type(err).__name__)
            return


def captured(fn):
    if mode == "tuner":
        return timing.graph_ms(fn, torch.device("cuda"), calls=400)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="global"):
        for _ in range(400):
            fn()


y = torch.randn(256, 256, device="cuda")
t = threading.Thread(target=writer)
t.start()
try:
    for _ in range(10):
        captured(lambda: y @ y)
except Exception as err:
    errors.append(type(err).__name__)
finally:
    stop.set()
    t.join()
print("ERRORS", len(errors))
"""


def _errors(mode: str) -> int:
    run = subprocess.run(
        [sys.executable, "-c", CHILD, mode], capture_output=True, text=True, timeout=600
    )
    line = [x for x in run.stdout.splitlines() if x.startswith("ERRORS")]
    return int(line[-1].split()[1]) if line else -1


@pytest.mark.budget(30)  # child processes
def test_tuner_capture_lets_other_threads_sync():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    def clean(mode):
        assert _errors(mode) == 0, mode

    clean("tuner")
    must_fail(lambda: clean("global"), "global-mode capture breaks the writer thread")
