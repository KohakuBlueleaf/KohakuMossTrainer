"""SAMPLE_AT_START and SAMPLE_COLUMNS: a callback with `at_start` fires once before a
fresh run's first step (not on resume, not without the flag), so sampling runs at step
0, every, 2 every, ...; the step sheet lays the images `columns` per row in order, each
row the next run of prompts."""

import pytest
from PIL import Image

from kohakumoss.inference.pipeline import batch_slices
from kohakumoss.training.build.loop import sample_settings
from kohakumoss.training.callbacks.evaluate import step_sheet, thumb
from kohakumoss.training.callbacks.runner import CallbackRunner
from kohakumoss.training.state import TrainState
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


class _Sampler:
    def __init__(self, every: int, at_start: bool):
        self.every = every
        self.at_start = at_start
        self.calls = []

    def on_step_end(self, state, out):
        self.calls.append(state.step)


def _fired(start: int, at_start: bool, runner_cls=CallbackRunner) -> list[int]:
    cb = _Sampler(4, at_start)
    runner = runner_cls([cb], start_step=start)
    state = TrainState(step=start)
    runner.start(state)
    for step in range(start + 1, 13):
        state.step = step
        runner.run(state, None)
    return cb.calls


def check_start(runner_cls) -> None:
    assert _fired(0, True, runner_cls) == [0, 4, 8, 12]
    assert _fired(0, False, runner_cls) == [4, 8, 12]
    assert _fired(4, True, runner_cls) == [8, 12]


def test_sampling_fires_at_step_zero():
    check_start(CallbackRunner)


def test_negative_start_ignored():
    class NoStart(CallbackRunner):
        def start(self, state):
            return None

    must_fail(lambda: check_start(NoStart), "no step-0 sample")


def _tiles(n: int) -> list:
    return [Image.new("RGB", (40 + 8 * i, 32), (10 * i, 0, 0)) for i in range(n)]


def check_sheet(sheet_fn) -> None:
    images = _tiles(16)
    sheet = sheet_fn(images, [f"p{i}" for i in range(16)], 8)
    widths = [max(images[j].width, images[j + 8].width) for j in range(8)]
    side = sheet.width - sum(widths)
    header = sheet.height - 2 * 32
    for i, im in enumerate(images):
        row, col = divmod(i, 8)
        x, y = side + sum(widths[:col]), header + 32 * row
        assert sheet.getpixel((x + 1, y + 1)) == im.getpixel((0, 0)), i


def test_step_sheet_rows():
    check_sheet(step_sheet)


def test_negative_step_sheet_one_row():
    def one_row(images, prompts, columns, side=0):
        return step_sheet(images, prompts, len(images), side)

    must_fail(lambda: check_sheet(one_row), "every image in one row")


def check_slices(fn) -> None:
    assert fn(16, 0) == [slice(0, 16)]
    assert fn(16, 8) == [slice(0, 8), slice(8, 16)]
    assert fn(5, 4) == [slice(0, 4), slice(4, 8)]
    assert fn(0, 0) == []


def test_batch_slices_zero_is_one_batch():
    check_slices(batch_slices)


def test_negative_batch_slices_zero_as_default():
    must_fail(lambda: check_slices(lambda n, b: batch_slices(n, b or 4)), "0 = 4")


def test_sample_batch_reaches_generate():
    k = {
        "SAMPLE_PROMPTS": ["a", "b"],
        "SAMPLE_SEEDS": (0,),
        "SAMPLE_SIZE": ((512, 768), (768, 512)),
        "SAMPLE_STEPS": 4,
        "SAMPLE_CFG": 3.0,
        "SAMPLE_SHIFT": 5.0,
        "SAMPLE_SKIP_LAYERS": (),
    }
    assert sample_settings(k)["batch"] == 4
    assert sample_settings({**k, "SAMPLE_BATCH": 0})["batch"] == 0
    assert sample_settings(k)["sizes"] == ((512, 768), (768, 512))


def test_thumb_long_side():
    assert thumb(Image.new("RGB", (1360, 768)), 512).size == (512, 289)
    assert thumb(Image.new("RGB", (300, 200)), 512).size == (300, 200)
