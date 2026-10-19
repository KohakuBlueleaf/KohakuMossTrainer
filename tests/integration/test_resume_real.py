"""Checkpoint IO and resume on real training runs (LoRA, the KOHAKUMOSS_DATASET dataset, captions
from an offline text cache, latents encoded per step, tuning off): a run saves complete
step directories; a run resumed from the step-2 checkpoint reproduces the
uninterrupted run, its first step bit for bit (same weights, optimizer, data and
noise) and its step-4 checkpoint (adapter, optimizer, schedule, scaler, counters,
loader position): plain values and integer tensors exactly, floating tensors as closely
as a second uninterrupted run does. With optim_state "cpu" the resumed CPUAdamW keeps
its masters and moments as host fp32 tensors and every update lane bound to the live
parameter group (so the schedule's lr reaches the update)."""

import gc
import json
import shutil
import statistics
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.memory.apply import MemoryEngines
from kohakumoss.offload.optim.cpu_optim import HOST_STATE, CPUAdamW
from kohakumoss.training.checkpoint.io import load_checkpoint
from kohakumoss.training.drivers import Parts
from kohakumoss.training.entry import run
from kohakumoss_data.cache.build import build_text
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import build_encoder, data_config
from tests._util.negative import must_fail
from tests.integration.conftest import real_knobs

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
STEPS = 4
SAVED = 2
LORA = {"ALGO": "lora", "DIM": 8, "ALPHA": 8.0, "LR": 5e-4}
SHORT = {"TARGET_AREA": 384 * 384, "MAX_TOKENS": 2048}
FULL_SWAP_CPU = {
    "ALGO": "full",
    "LR": 5e-4,
    "MEMORY_PROFILE": "ckpt-stream2-optcpu",
    "EXPORT_FORMATS": (),
}
NOISE = 1.5  # see docs/internals/fp16.md#real-model-structural-tests
FILES = {"trainable.safetensors", "optim.pt", "rng.0.pt", "state.json"}


@pytest.fixture(scope="module")
def text_cache(weights_root, dataset_slice, tmp_path_factory) -> Path:
    """The offline text cache of the slice (every caption variant, the real encoder)."""
    cache = tmp_path_factory.mktemp("text_cache")
    k = real_knobs(weights_root, dataset_slice, cache, CACHE_DIR=str(cache), **SHORT)
    encode = build_encoder(
        "kohakumoss.encoders.text_encoder",
        {
            "path": str(weights_root / "base" / "text_encoder"),
            "device": "cuda",
            "plan": {"memory": "stream"},
        },
    )
    cfg = data_config(
        k,
        latent_cache="off",
        text_cache="offline",
        text_encoder_spec=encode.encoder_spec,
    )
    build_text(MossDataset(cfg), encode, 64)
    del encode
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def _knobs(weights_root, dataset, cache: Path, out: Path, **kw) -> dict:
    """The short LoRA resume run; `kw` overrides any of its knobs."""
    run_knobs = dict(
        CACHE_DIR=str(cache),
        TEXT_CACHE="offline",
        LATENT_CACHE="off",
        VAE_ENCODE={"compile": "off"},
        STEPS=STEPS,
        SAVE_EVERY=SAVED,
        KEEP_LAST=0,
        SCHEDULE={"mode": "cosine", "warmup": 1, "end": -1},
        TUNE="off",
        **SHORT,
        **LORA,
    )
    return real_knobs(weights_root, dataset, out, **{**run_knobs, **kw})


def _rows(out: Path) -> dict[int, dict]:
    text = (out / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    return {r["step"]: r for r in rows}


def _ckpt(out: Path, step: int) -> Path:
    return out / "run" / "checkpoints" / f"step_{step:08d}"


def check_layout(path: Path, step: int) -> None:
    """A complete step directory: every part on disk, state.json at `step`."""
    names = {p.name for p in path.iterdir()} if path.is_dir() else set()
    assert FILES <= names, sorted(names)
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    assert state["step"] == step, state["step"]


def check_first_step(
    straight: dict[int, dict],
    resumed: dict[int, dict],
    saved: int = SAVED,
    steps: int = STEPS,
) -> None:
    """The resumed run logs steps saved+1..steps; its first step's loss is the
    uninterrupted run's, exactly."""
    assert sorted(resumed) == list(range(saved + 1, steps + 1)), sorted(resumed)
    first = saved + 1
    assert resumed[first]["loss"] == straight[first]["loss"], (
        resumed[first]["loss"],
        straight[first]["loss"],
    )


def leaves(obj, path: tuple = ()) -> dict[tuple, object]:
    """Every leaf of nested dicts / lists / tuples by its key path."""
    match obj:
        case dict():
            items = obj.items()
        case list() | tuple():
            items = enumerate(obj)
        case _:
            return {path: obj}
    out = {}
    for key, value in items:
        out.update(leaves(value, (*path, key)))
    return out


def _rel(a: Tensor, b: Tensor) -> float:
    d = (a.double() - b.double()).norm()
    return (d / b.double().norm().clamp_min(1e-30)).item()


def kind(key: tuple) -> tuple[str, str]:
    """A checkpoint leaf's kind: its part and its name without the module path
    (lora_up.weight, alpha, exp_avg, ...)."""
    return str(key[0]), str(key[-1]).split(".", 1)[-1]


def kind_rel(straight: dict, other: dict) -> dict[tuple[str, str], float]:
    """Per kind of floating tensor: the median relative difference of `other`'s leaves
    from `straight`'s (both flattened by `leaves`)."""
    rels: dict[tuple[str, str], list[float]] = {}
    for key, x in straight.items():
        if isinstance(x, Tensor) and x.is_floating_point():
            rels.setdefault(kind(key), []).append(_rel(other[key], x))
    return {k: statistics.median(v) for k, v in rels.items()}


def check_state(straight: dict, resumed: dict, rerun: dict) -> None:
    """Same parts, keys, shapes and dtypes; integer tensors and plain values equal;
    per kind of floating tensor, the resumed run's median relative difference from the
    straight run at most NOISE x a second straight run's (exact where that is)."""
    a, b, c = leaves(straight), leaves(resumed), leaves(rerun)
    assert a.keys() == b.keys() == c.keys(), sorted(map(str, set(a) ^ set(b)))[:8]
    for key, x in a.items():
        y = b[key]
        if not isinstance(x, Tensor):
            assert x == y, (key, x, y)
            continue
        assert isinstance(y, Tensor) and (x.shape, x.dtype) == (y.shape, y.dtype), key
        if not x.is_floating_point():
            assert torch.equal(x, y), key
    noise = kind_rel(a, c)
    for k, rel in kind_rel(a, b).items():
        assert rel <= NOISE * noise[k], (k, rel, noise[k])


def _forget_step_rng(original):
    """Parts.restore that keeps the step generators' fresh state."""

    def restore(self, ckpt, state):
        original(self, {**ckpt, "step_rng": self.step.rng_state()}, state)

    return restore


def test_resume_reproduces_the_uninterrupted_run(
    weights_root, dataset_slice, text_cache, tmp_path, monkeypatch
):
    def knobs(out: Path, **kw) -> dict:
        return _knobs(weights_root, dataset_slice, text_cache, out, **kw)

    straight, rerun, resumed, forgot = (
        tmp_path / n for n in ("straight", "rerun", "resumed", "forgot")
    )
    assert run(knobs(straight)).step == STEPS
    for step in (SAVED, STEPS):
        check_layout(_ckpt(straight, step), step)
    must_fail(lambda: check_layout(_ckpt(straight, SAVED), STEPS), "step 2 read as 4")
    assert run(knobs(rerun)).step == STEPS
    again = load_checkpoint(_ckpt(rerun, STEPS))
    resume = str(_ckpt(straight, SAVED))
    assert run(knobs(resumed, RESUME=resume)).step == STEPS
    check_first_step(_rows(straight), _rows(resumed))
    want = load_checkpoint(_ckpt(straight, STEPS))
    check_state(want, load_checkpoint(_ckpt(resumed, STEPS)), again)
    monkeypatch.setattr(Parts, "restore", _forget_step_rng(Parts.restore))
    run(knobs(forgot, RESUME=resume))
    must_fail(
        lambda: check_first_step(_rows(straight), _rows(forgot)),
        "a resume without the step generators: first step",
    )
    must_fail(
        lambda: check_state(want, load_checkpoint(_ckpt(forgot, STEPS)), again),
        "a resume without the step generators: step-4 state",
    )


def host_snapshot(opt: CPUAdamW) -> dict:
    """(device type, dtype) of every host-state tensor, and whether each parameter's
    update lane reads the live parameter group."""
    state = {}
    for gi, group in enumerate(opt.param_groups):
        for pi, p in enumerate(group["params"]):
            for key in HOST_STATE:
                t = opt.state[p].get(key)
                state[gi, pi, key] = None if t is None else (t.device.type, t.dtype)
    bound = [
        opt._lanes[id(p)].group is group
        for group in opt.param_groups
        for p in group["params"]
    ]
    return {"state": state, "bound": bound}


def check_host_state(snap: dict) -> None:
    assert snap["state"] and snap["bound"], snap
    off = {k: v for k, v in snap["state"].items() if v != ("cpu", torch.float32)}
    assert not off, list(off.items())[:4]
    assert all(snap["bound"]), f"{snap['bound'].count(False)} lanes on a stale group"


def test_cpu_adamw_resume(
    weights_root, dataset_slice, text_cache, tmp_path, monkeypatch
):
    def knobs(out: Path, **kw) -> dict:
        return _knobs(
            weights_root,
            dataset_slice,
            text_cache,
            out,
            MEMORY_PROFILE="ckpt-resident-optcpu",
            **kw,
        )

    loads = []
    original = CPUAdamW.load_state_dict

    def spying(self, state_dict):
        original(self, state_dict)
        loads.append((self, host_snapshot(self)))

    monkeypatch.setattr(CPUAdamW, "load_state_dict", spying)
    straight, rerun, resumed = (tmp_path / n for n in ("straight", "rerun", "resumed"))
    assert run(knobs(straight)).step == STEPS
    assert run(knobs(rerun)).step == STEPS
    resume = str(_ckpt(straight, SAVED))
    assert run(knobs(resumed, RESUME=resume)).step == STEPS
    assert len(loads) == 1, len(loads)
    opt, snap = loads[0]
    check_host_state(snap)
    check_first_step(_rows(straight), _rows(resumed))
    check_state(
        load_checkpoint(_ckpt(straight, STEPS)),
        load_checkpoint(_ckpt(resumed, STEPS)),
        load_checkpoint(_ckpt(rerun, STEPS)),
    )
    saved = load_checkpoint(_ckpt(straight, SAVED))["optim"]["optimizer"]
    Optimizer.load_state_dict(opt, saved)
    must_fail(
        lambda: check_host_state(host_snapshot(opt)),
        "torch's load: state cast to the parameters' device, lanes on the old groups",
    )


def _no_refresh(self) -> None:
    return None


def test_swap_cpu_adamw_full_resume(
    weights_root, dataset_slice, text_cache, tmp_path, monkeypatch
):
    """Full FT, swap-streamed blocks, host AdamW: the resumed run's first step reads
    the restored masters (the streaming copies refreshed), so its loss is the
    uninterrupted run's. The straight run's final checkpoint (every master and moment of
    the backbone) is removed once written and the resumed runs write none
    (SAVE_EVERY=0); only the step-2 one stays on disk."""
    steps = SAVED + 1

    def knobs(out: Path, **kw) -> dict:
        return _knobs(
            weights_root,
            dataset_slice,
            text_cache,
            out,
            **{**FULL_SWAP_CPU, "STEPS": steps, **kw},
        )

    straight, resumed, stale = (tmp_path / n for n in ("straight", "resumed", "stale"))
    assert run(knobs(straight)).step == steps
    shutil.rmtree(_ckpt(straight, steps))
    resume = str(_ckpt(straight, SAVED))
    assert run(knobs(resumed, RESUME=resume, SAVE_EVERY=0)).step == steps
    assert not _ckpt(resumed, steps).exists()
    check_first_step(_rows(straight), _rows(resumed), SAVED, steps)
    monkeypatch.setattr(MemoryEngines, "after_restore", _no_refresh)
    run(knobs(stale, RESUME=resume, SAVE_EVERY=0))
    must_fail(
        lambda: check_first_step(_rows(straight), _rows(stale), SAVED, steps),
        "resume without refreshing the streaming copies from the restored masters",
    )
