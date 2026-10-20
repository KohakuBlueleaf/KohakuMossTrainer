"""VAE_DTYPE: the knob rule (auto picks bf16 where the device has it, never an unsafe
fp16), the dtype reaching the VAE in training and in the cache encoder, and the
same-size runs of a batched encode (model-free)."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.encoders import vae as vae_mod
from kohakumoss.encoders.vae import (
    FluxVae,
    _runs,
    resolve_vae_dtype,
    vae_dtype_problems,
)
from kohakumoss.offload.encoders.vae import VaePlan, compiles
from kohakumoss.offload.encoders.vae import plan_problems as vae_plan_problems
from kohakumoss.training.build import model as model_build
from kohakumoss.training.knobs import validate
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")
UNSAFE = "fp16 decode overflows"


def _has_bf16(monkeypatch, value: bool):
    monkeypatch.setattr(vae_mod, "_has_bf16", lambda device: value)


def test_auto_dtype(monkeypatch):
    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", UNSAFE)
    assert resolve_vae_dtype("auto", CPU) == torch.bfloat16
    assert resolve_vae_dtype("fp32", CPU) == torch.float32
    _has_bf16(monkeypatch, True)
    assert resolve_vae_dtype("auto", "cuda:0") == torch.bfloat16
    _has_bf16(monkeypatch, False)
    assert resolve_vae_dtype("auto", "cuda:0") == torch.float32
    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", None)
    assert resolve_vae_dtype("auto", "cuda:0") == torch.float16
    for name, dtype in vae_mod.DTYPES.items():
        assert resolve_vae_dtype(name, CPU) == dtype


def test_auto_never_picks_unsafe_fp16(monkeypatch):
    def check():
        assert resolve_vae_dtype("auto", "cuda:0") != torch.float16

    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", UNSAFE)
    _has_bf16(monkeypatch, False)
    check()
    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", None)
    must_fail(check, "fp16 picked while it is marked unsafe")


def test_dtype_rule(monkeypatch):
    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", UNSAFE)
    assert vae_dtype_problems("auto", CPU) == []
    assert any("not in" in m for m in vae_dtype_problems("int8", CPU))
    assert any(UNSAFE in m for m in vae_dtype_problems("fp16", CPU))
    _has_bf16(monkeypatch, False)
    assert any("no bf16" in m for m in vae_dtype_problems("bf16", "cuda:0"))
    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", None)
    assert vae_dtype_problems("fp16", CPU) == []


def test_training_knob_reports_the_rule(monkeypatch):
    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", UNSAFE)
    assert any(UNSAFE in m for m in validate_vae(runnable(VAE_DTYPE="fp16")))
    assert validate_vae(runnable()) == []


def validate_vae(k: dict) -> list[str]:
    return [m for m in validate(k) if "VAE_DTYPE" in m]


def _captured_training_dtype(monkeypatch, name: str) -> torch.dtype:
    seen = {}

    def load(path, device="cuda", dtype=torch.float32, mode="mode"):
        seen["dtype"] = dtype
        return SimpleNamespace()

    monkeypatch.setattr(model_build.FluxVae, "from_pretrained", load)
    placed = SimpleNamespace(plan=VaePlan())
    monkeypatch.setattr(model_build, "place_vae", lambda vae, *a, **kw: placed)
    k = runnable(
        WEIGHTS="weights",
        VAE="vae",
        LATENT_CACHE="off",
        TEXT_CACHE="offline",
        SAMPLE_EVERY=0,
        VAE_DTYPE=name,
    )
    model_build.build_encoders(k, CPU, None)
    return seen["dtype"]


def test_training_vae_gets_the_knob_dtype(monkeypatch):
    def check():
        assert _captured_training_dtype(monkeypatch, "fp32") == torch.float32
        assert _captured_training_dtype(monkeypatch, "auto") == torch.bfloat16

    check()
    monkeypatch.setattr(model_build, "resolve_vae_dtype", lambda n, d: torch.float32)
    must_fail(check, "the knob is ignored")


def test_cache_encoder_gets_the_dtype_and_placement(monkeypatch):
    seen = {}

    def load(path, device="cuda", dtype=torch.float32, mode="mode"):
        seen["dtype"] = dtype
        return SimpleNamespace()

    def place(vae, spec, device):
        seen["spec"] = spec
        return SimpleNamespace(encode_images=None)

    monkeypatch.setattr(FluxVae, "from_pretrained", load)
    monkeypatch.setattr(vae_mod, "place_vae", place)
    encode = vae_mod.vae_encoder("vae", CPU, "bf16")
    assert seen["dtype"] == torch.bfloat16 and encode.encoder_spec == "bf16"
    assert seen["spec"]["memory"] == "auto"
    vae_mod.vae_encoder("vae", CPU, "fp32", plan={"memory": "offload", "tile": True})
    assert seen["dtype"] == torch.float32
    assert seen["spec"] == {"memory": "offload", "batch": 1, "tile": True}


def assert_host_vae_keys_like_the_run(encoder, monkeypatch) -> None:
    """A host VAE (memory 'cpu') for an fp16 card loads, computes and keys in the
    card's auto dtype, like the run that reads its latents."""
    seen = {}

    def load(path, device="cuda", dtype=torch.float32, mode="mode"):
        seen["dtype"] = dtype
        return SimpleNamespace()

    monkeypatch.setattr(FluxVae, "from_pretrained", load)
    monkeypatch.setattr(
        vae_mod, "place_vae", lambda vae, spec, d: SimpleNamespace(encode_images=None)
    )
    _has_bf16(monkeypatch, False)
    monkeypatch.setattr(vae_mod, "FP16_PROBLEM", None)
    encode = encoder("vae", "cuda:0", "auto", {"memory": "cpu"})
    run = resolve_vae_dtype("auto", "cuda:0")
    assert seen["dtype"] == run == torch.float16 and encode.encoder_spec == "fp16"


def _placement_dtype(path, device, dtype, plan):
    where = CPU if plan.get("memory") == "cpu" else device
    return vae_mod.vae_encoder(path, device, resolve_vae_dtype(dtype, where), plan=plan)


def test_a_host_vae_keys_like_the_run(monkeypatch):
    assert_host_vae_keys_like_the_run(
        lambda path, device, dtype, plan: vae_mod.vae_encoder(
            path, device, dtype, plan=plan
        ),
        monkeypatch,
    )
    must_fail(
        lambda: assert_host_vae_keys_like_the_run(_placement_dtype, monkeypatch),
        "auto resolved for the host the VAE computes on",
    )


def test_compile_rule():
    cuda = torch.device("cuda", 0)

    def check(rule):
        assert rule(VaePlan(), cuda)
        assert not rule(VaePlan(), CPU)
        assert not rule(VaePlan(memory="offload"), cuda)
        assert not rule(VaePlan(tile=True), cuda)
        assert not rule(VaePlan(compile="off"), cuda)
        assert rule(VaePlan(compile="on", memory="offload"), cuda)

    check(compiles)
    must_fail(lambda: check(lambda plan, d: d.type == "cuda"), "compile any placement")
    assert any("compile" in m for m in vae_plan_problems(VaePlan(compile="yes")))


def test_runs_respect_sizes_and_budget(monkeypatch):
    sizes = [(4, 4)] * 3 + [(8, 8)] * 3

    def check(runs):
        for lo, hi in runs:
            assert len(set(sizes[lo:hi])) == 1
            assert (hi - lo) * 256 * sizes[lo][0] * sizes[lo][1] <= max(
                256 * 4 * 4 * 3, 256 * sizes[lo][0] * sizes[lo][1]
            )

    check(_runs(sizes, 8, 256 * 4 * 4 * 3))
    must_fail(lambda: check([(0, 6)]), "a run across sizes")
