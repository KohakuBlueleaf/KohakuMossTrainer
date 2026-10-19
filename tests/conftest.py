"""Test configuration: hardware-marker skipping, deterministic seeds, kernel tuning off
against an empty session kernel cache (each op runs its default config, no host table is
read; LyCORIS's tiling likewise), the kernel backends loaded once at session start (so a
test's fake backend checks are not overwritten by a first build mid-test), and a
per-test time budget.

The budget fails any test whose call phase runs longer than KOHAKUMOSS_TEST_BUDGET
seconds (default BUDGET; a `budget(seconds)` marker overrides it). Tests marked `slow`
or `real_weights` (the whole-integration runs) have no budget, and
KOHAKUMOSS_TEST_BUDGET=0 disables every budget, markers included, for a first run that
compiles the kernels. A
failed test's frames drop their locals once it is reported.

Environment variables the suite reads:
- KOHAKUMOSS_WEIGHTS: the weights root of the real-weights tests (base/, fp16fix/).
- KOHAKUMOSS_DATASET: a kohya-layout dataset the real-weights tests slice.
- KOHAKUMOSS_FP16_FIX_FILE: the fp16-fix DiT file (default <weights>/fp16fix/...).
- KOHAKUMOSS_TEST_ADAPTER: an adapter file the inference / grid tests use instead of
  training one in a 2-step run.
- KOHAKUMOSS_TEST_TEXT_EMBEDS: a safetensors file of cached text embeddings (keys "0",
  "1", ...) the real-weights training test reads instead of running the text encoder.
- KOHAKUMOSS_REPRO_STEP: the recorded failing-step file test_repro_step_nan replays.
- KOHAKUMOSS_TEST_DEVICE: the device the kernel tests run on (e.g. xpu:1).
- KOHAKUMOSS_TEST_BUDGET, KOHAKUMOSS_PRECISION_DIR (error-statistics tables),
  KOHAKUMOSS_KERNEL_CACHE, KOHAKUMOSS_LOG_LEVEL."""

import logging
import os
import shutil
import sys
import tempfile
import time
import traceback

import pytest
import torch

from kohakumoss.adapters.lycoris.tuning import set_lycoris_tuning
from kohakumoss_common import log as klog
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.registry import load_backends
from kohakumoss_kernels.tuning import decisions, tuner, window

pytest_plugins = ["tests._util.synthetic_source"]
SM_MARKERS = {"sm70": 70, "sm75": 75, "sm86": 86, "sm89": 89, "sm120": 120}
BUDGET = 10.0
UNBUDGETED = {"slow", "real_weights"}
SESSION_CACHE: list[str] = []


def pytest_configure(config):
    config.addinivalue_line("markers", "budget(seconds): this test's time budget")
    root = tempfile.mkdtemp(prefix="kohakumoss-test-cache-")
    SESSION_CACHE.append(root)
    os.environ["KOHAKUMOSS_KERNEL_CACHE"] = os.path.join(root, "kohakumoss")
    os.environ["LYCORIS_KERNEL_CACHE_DIR"] = os.path.join(root, "lycoris")
    tuner._TABLES.clear()
    tuner.set_mode("off")
    set_lycoris_tuning(False)
    load_backends()


def pytest_unconfigure(config):
    for root in SESSION_CACHE:
        shutil.rmtree(root, ignore_errors=True)


def _budget(item) -> float:
    """The call-phase budget of `item` in seconds (0: none): KOHAKUMOSS_TEST_BUDGET=0
    disables every budget, markers included; else its `budget` marker, else the env
    value, else BUDGET."""
    env = os.environ.get("KOHAKUMOSS_TEST_BUDGET")
    marks = {m.name for m in item.iter_markers()}
    if marks & UNBUDGETED or (env is not None and float(env) == 0):
        return 0.0
    mark = item.get_closest_marker("budget")
    if mark is not None:
        return float(mark.args[0])
    return float(env if env is not None else BUDGET)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    start = time.perf_counter()
    out = yield
    took = time.perf_counter() - start
    budget = _budget(item)
    if budget and took > budget:
        pytest.fail(
            f"{item.nodeid} took {took:.1f} s, over its {budget:.0f} s budget "
            "(model-free tests must be fast; a first kernel compile runs with "
            "KOHAKUMOSS_TEST_BUDGET=0)",
            pytrace=False,
        )
    return out


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Once a failed phase is reported, its frames drop their locals: a failing test's
    tensors (a real model) do not stay alive for every later test."""
    report = yield
    if call.excinfo is not None:
        traceback.clear_frames(call.excinfo.tb)
    return report


def _cuda_arch():
    if not torch.cuda.is_available():
        return None
    return arch_of("cuda")


def pytest_collection_modifyitems(config, items):
    arch = _cuda_arch()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for item in items:
        marks = {m.name for m in item.iter_markers()}
        if "cuda" in marks and arch is None:
            item.add_marker(pytest.mark.skip(reason="no CUDA GPU"))
        if "gpu" in marks and arch is None and not has_xpu:
            item.add_marker(pytest.mark.skip(reason="no CUDA or XPU device"))
        for name, sm in SM_MARKERS.items():
            if name in marks and (arch is None or arch.sm != sm):
                item.add_marker(pytest.mark.skip(reason=f"needs {name}"))
        if "xpu" in marks and not has_xpu:
            item.add_marker(pytest.mark.skip(reason="no XPU"))
        if "windows" in marks and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(reason="Windows only"))
        if "multi_gpu" in marks and n_gpu < 2:
            item.add_marker(pytest.mark.skip(reason="needs two GPUs"))
        if "real_weights" in marks and not os.environ.get("KOHAKUMOSS_WEIGHTS"):
            item.add_marker(pytest.mark.skip(reason="KOHAKUMOSS_WEIGHTS unset"))


@pytest.fixture(autouse=True, scope="session")
def _basetemp(tmp_path_factory):
    """pytest's base temp directory, made before any test patches sys.platform."""
    tmp_path_factory.getbasetemp()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


@pytest.fixture(autouse=True)
def _tune_off():
    """Each test starts with TUNE="off" (no measurement), an open tuning window and no
    recorded decisions."""
    tuner.set_mode("off")
    window.reopen()
    decisions.clear()
    yield
    tuner.set_mode("off")
    window.reopen()
    decisions.clear()


@pytest.fixture(autouse=True)
def _logging_roots():
    """Entry points under test call log.setup(); each test starts unconfigured
    (records propagate to pytest's capture, library floors and logging.disable as the
    test found them)."""
    floors = {n: logging.getLogger(n).level for n in klog.NOISY_LIBRARY_LEVELS}
    disabled = logging.root.manager.disable
    yield
    logging.disable(disabled)
    for name, level in floors.items():
        logging.getLogger(name).setLevel(level)
    for name in klog.ROOTS:
        root = logging.getLogger(name)
        for handler in [h for h in root.handlers if getattr(h, klog.MARK, False)]:
            root.removeHandler(handler)
            handler.close()
        root.propagate = True
        root.setLevel(logging.NOTSET)
    setattr(logging.getLogger(klog.ROOTS[0]), klog.COLOUR, False)


@pytest.fixture
def device():
    """`KOHAKUMOSS_TEST_DEVICE` (e.g. xpu:1) when set, else CUDA, XPU, CPU."""
    if os.environ.get("KOHAKUMOSS_TEST_DEVICE"):
        return torch.device(os.environ["KOHAKUMOSS_TEST_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


@pytest.fixture
def weights_dir():
    return os.environ.get("KOHAKUMOSS_WEIGHTS")
