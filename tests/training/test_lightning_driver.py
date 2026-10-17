"""DRIVER='lightning' trains what the native driver trains: same micro-batch arming
(fused per-parameter steps, gradient sync contexts), same parameters after N steps."""

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.training.entry import run
from tests._util.negative import must_fail
from tests.training.test_entry_smoke import smoke_knobs

pytestmark = [pytest.mark.cpu, pytest.mark.slow]
pytest.importorskip("lightning")


def _trained(tmp_path, name: str, **kw) -> dict:
    run(smoke_knobs(tmp_path / name, STEPS=2, SAVE_EVERY=0, GRAD_CLIP=0.0, **kw))
    ck = tmp_path / name / "smoke" / "checkpoints" / "step_00000002"
    return load_file(str(ck / "trainable.safetensors"))


def _moved(state: dict) -> None:
    ups = [v for k, v in state.items() if "lora_up" in k]
    assert ups and max(float(v.abs().max()) for v in ups) > 0


def _same(a: dict, b: dict) -> None:
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=1e-7, msg=key)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"MEMORY_OVERRIDES": {"fused_step": True}}],
    ids=["plain", "fused_step"],
)
def test_lightning_matches_native(tmp_path, overrides):
    native = _trained(tmp_path, "native", GRAD_ACC=2, **overrides)
    light = _trained(tmp_path, "light", GRAD_ACC=2, DRIVER="lightning", **overrides)
    _moved(native)
    _moved(light)
    _same(native, light)
    frozen = {k: torch.zeros_like(v) if "lora_up" in k else v for k, v in light.items()}
    must_fail(lambda: _moved(frozen), "nothing trained")
