"""Real export / merge scripts on a 2-step LoRA run of the real weights (the card's
16-bit precision: bf16 where the card has it, else fp16): the export script on the run's checkpoint writes
the same tensors as the run's own export (comfy and LyCORIS), and so does every
checkpoint the session's earlier real runs left (their algos / precisions / fp16 fixes
recorded, at the dtype of their export's weight tensors); a checkpoint stripped of its
spec / fold metadata is refused and nothing is written; a FUSED_QKV comfy file merges
into the same transformer as the unfused one; a merge into an fp16-fix file (one this
test folds with pre-gate scales, and the shipped one) adds the plain merge's delta W
times the fold's row scales, within each base dtype's rounding of delta W; the run
directory holds its log.txt."""

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest
import torch
from kohakuengine import run as run_script
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kohakumoss.model import load_backbone
from kohakumoss.model.io.source import checkpoint_files
from kohakumoss.model.numerics import output_row_scales, unfold
from kohakumoss.model.numerics.fold import fold_rows
from kohakumoss.training.entry import run
from tests._util.negative import must_fail
from tests.integration.conftest import real_knobs

META_KEYS = ("kohakumoss_spec", "kohakumoss_fold")
FIELD = r"kohakumoss_(?:spec|fold)"
REFUSAL = (
    rf": training-state metadata without {FIELD}(?:, {FIELD})?; "
    r"re-export from a run of this version$"
)
FP32_EPS = torch.finfo(torch.float32).eps

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "scripts" / "export" / "export.py"
MERGE = ROOT / "scripts" / "export" / "merge.py"
FIX_SCRIPT = ROOT / "scripts" / "export" / "fp16_fix.py"
BUILT_FIX = {"C": 0.0625, "S_ATTN": 0.5, "S_FFN": 0.5}
STORED = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


@pytest.fixture(scope="module")
def trained(weights_root, dataset_slice, tmp_path_factory) -> Path:
    """<out>/run of a 2-step LoRA run (the card's precision, `real_knobs`) with its
    step-2 checkpoint and comfy + LyCORIS exports."""
    out = tmp_path_factory.mktemp("export_run")
    k = real_knobs(
        weights_root,
        dataset_slice,
        out,
        STEPS=2,
        SAVE_EVERY=2,
        KEEP_LAST=1,
        ALGO="lora",
        DIM=8,
        ALPHA=8.0,
        LR=1e-2,
        EXPORT_FORMATS=("lycoris", "comfy"),
    )
    run(k)
    return out / "run"


def _state(run_dir: Path) -> Path:
    """The run's latest checkpoint state (the one its end-of-run export was taken from)."""
    return sorted((run_dir / "checkpoints").glob("step_*/trainable.safetensors"))[-1]


def check_same(a: dict, b: dict) -> None:
    """Same keys, bit-identical tensors."""
    assert sorted(a) == sorted(b), sorted(set(a) ^ set(b))[:6]
    for key in a:
        assert torch.equal(a[key], b[key]), key


def check_close(a: dict, b: dict, rel: float) -> None:
    """Same keys; every tensor within `rel` (relative max-abs) of the other."""
    assert sorted(a) == sorted(b), sorted(set(a) ^ set(b))[:6]
    for key in a:
        x, y = a[key].float(), b[key].float()
        bound = rel * max(float(y.abs().max()), 1e-12)
        assert float((x - y).abs().max()) <= bound, key


@pytest.mark.parametrize("fmt", ["comfy", "lycoris"])
def test_export_script_matches_the_run_export(fmt, trained, weights_root, tmp_path):
    out = tmp_path / f"{fmt}.safetensors"
    own = load_file(str(next((trained / "export").glob(f"*-{fmt}.safetensors"))))
    sets = {
        "WEIGHTS": str(weights_root / "base"),
        "ADAPTER": str(_state(trained)),
        "OUT": str(out),
        "FORMAT": fmt,
        "SAVE_DTYPE": weight_dtype(own),
        "DEVICE": "cuda",
    }
    run_script(str(EXPORT), set_overrides=sets)
    got = load_file(str(out))
    check_same(got, own)
    key = next(iter(own))
    off = {**own, key: own[key] * 2}
    must_fail(lambda: check_same(got, off), "an export with one tensor scaled")


def test_fused_qkv_merges_like_unfused(trained, weights_root, tmp_path):
    merged = {}
    for fused in (False, True):
        comfy = tmp_path / f"comfy_{fused}.safetensors"
        run_script(
            str(EXPORT),
            set_overrides={
                "WEIGHTS": str(weights_root / "base"),
                "ADAPTER": str(_state(trained)),
                "OUT": str(comfy),
                "FORMAT": "comfy",
                "FUSED_QKV": str(fused),
                "DEVICE": "cuda",
            },
        )
        out = tmp_path / f"merged_{fused}.safetensors"
        run_script(
            str(MERGE),
            set_overrides={
                "WEIGHTS": str(weights_root / "base"),
                "ADAPTERS": repr([(str(comfy), 1.0)]),
                "OUT": str(out),
                "SAVE_DTYPE": "fp32",
                "DEVICE": "cuda",
            },
        )
        merged[fused] = load_file(str(out))
        comfy.unlink()
    check_close(merged[True], merged[False], rel=1e-3)
    key = next(k for k in merged[False] if merged[False][k].numel() > 1)
    off = {**merged[False], key: merged[False][key] * 1.01 + 1.0}
    must_fail(lambda: check_close(merged[True], off, rel=1e-3), "a shifted tensor")


def _export(
    weights: Path, state: Path, out: Path, fmt: str = "comfy", dtype: str = "bf16"
) -> None:
    sets = {
        "WEIGHTS": str(weights),
        "ADAPTER": str(state),
        "OUT": str(out),
        "FORMAT": fmt,
        "SAVE_DTYPE": dtype,
        "DEVICE": "cuda",
    }
    run_script(str(EXPORT), set_overrides=sets)


def _stripped_run(trained: Path, root: Path) -> Path:
    """A copy of the run whose checkpoint state has no spec / fold metadata."""
    copy = root / "old_run"
    shutil.copytree(trained, copy, ignore=shutil.ignore_patterns("export"))
    state = _state(copy)
    with safe_open(str(state), "pt") as f:
        meta = {k: v for k, v in (f.metadata() or {}).items() if k not in META_KEYS}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    save_file(tensors, str(state), metadata=meta)
    return copy


def check_refused(weights: Path, state: Path, out: Path, dtype: str = "bf16") -> None:
    """Export of a state without its spec / fold metadata stops with the refusal and
    writes nothing."""
    try:
        _export(weights, state, out, dtype=dtype)
    except SystemExit as err:
        expected = rf"^export: {re.escape(str(state))}{REFUSAL}"
        assert re.match(expected, str(err)), str(err)
    else:
        raise AssertionError(f"export of {state} was not refused")
    assert not out.exists(), out


def test_export_of_a_checkpoint_without_metadata(trained, weights_root, tmp_path):
    old = _stripped_run(trained, tmp_path)
    with safe_open(str(_state(old)), "pt") as f:
        assert not set(META_KEYS) & set(f.metadata() or {})
    assert (old / "config.json").is_file()
    own = load_file(str(next((trained / "export").glob("*-comfy.safetensors"))))
    dtype = weight_dtype(own)
    old_out = tmp_path / "old.safetensors"
    check_refused(weights_root / "base", _state(old), old_out, dtype)
    out = tmp_path / "intact.safetensors"
    must_fail(
        lambda: check_refused(weights_root / "base", _state(trained), out, dtype),
        "a checkpoint with its metadata intact",
    )
    check_same(load_file(str(out)), own)


DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def weight_dtype(state: dict) -> str:
    """SAVE_DTYPE name of an export: its weight tensors' dtype (alphas are always fp32)."""
    return DTYPE_NAMES[
        next(t.dtype for k, t in state.items() if not k.endswith(".alpha"))
    ]


def session_runs(root: Path) -> list[Path]:
    """Run dirs under `root` (this session's base temp: every real run of the earlier
    tests) with a config.json, a checkpoint state and an end-of-run comfy export."""
    return [
        cfg.parent
        for cfg in sorted(root.rglob("config.json"))
        if list(cfg.parent.glob("checkpoints/step_*/trainable.safetensors"))
        and list((cfg.parent / "export").glob("*-comfy.safetensors"))
    ]


def test_every_session_checkpoint_exports_like_its_run(
    trained, tmp_path_factory, tmp_path, record_property
):
    runs = session_runs(tmp_path_factory.getbasetemp())
    kinds = []
    for i, run_dir in enumerate(runs):
        knobs = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        kinds.append(
            {k: knobs.get(k) for k in ("ALGO", "PRECISION", "FP16_FIX", "WEIGHTS")}
            | {"run": run_dir.name, "path": str(run_dir)}
        )
        own = load_file(str(next((run_dir / "export").glob("*-comfy.safetensors"))))
        dtype = weight_dtype(own)
        weights = Path(knobs["WEIGHTS"])
        out = tmp_path / f"r{i}.safetensors"
        _export(weights, _state(run_dir), out, dtype=dtype)
        check_same(load_file(str(out)), own)
        if i == 0:
            other = "fp32" if dtype != "fp32" else "bf16"
            wrong = tmp_path / f"r{i}_{other}.safetensors"
            _export(weights, _state(run_dir), wrong, dtype=other)
            must_fail(
                lambda: check_same(load_file(str(wrong)), own),
                f"a re-export at {other} instead of the export's {dtype}",
            )
        old = _stripped_run(run_dir, tmp_path / f"r{i}")
        check_refused(weights, _state(old), tmp_path / f"r{i}_old.safetensors", dtype)
    record_property("session_checkpoints", json.dumps(kinds))
    print("session checkpoints:", json.dumps(kinds))
    assert runs, "no real run of this session left a checkpoint and an export"


def _merge(weights: Path, adapter: Path, out: Path) -> None:
    sets = {
        "WEIGHTS": str(weights),
        "ADAPTERS": repr([(str(adapter), 1.0)]),
        "OUT": str(out),
        "SAVE_DTYPE": "fp32",
        "DEVICE": "cuda",
    }
    run_script(str(MERGE), set_overrides=sets)


def _lycoris(trained: Path) -> Path:
    return next((trained / "export").glob("*-lycoris.safetensors"))


def stored_dtype(path: Path) -> torch.dtype:
    """The dtype most of a checkpoint's floating tensors are stored in."""
    counts = Counter()
    for f in checkpoint_files(path):
        with safe_open(str(f), "pt") as h:
            names = (h.get_slice(k).get_dtype() for k in h.keys())
            counts.update(STORED[n] for n in names if n in STORED)
    return counts.most_common(1)[0][0]


@pytest.fixture(scope="module")
def plain_merge(trained, weights_root, tmp_path_factory) -> Path:
    """The run's LyCORIS export merged into the base weights (fp32)."""
    out = tmp_path_factory.mktemp("plain_merge") / "merged.safetensors"
    _merge(weights_root / "base", _lycoris(trained), out)
    return out


@pytest.fixture(scope="module")
def built_fix(weights_root, tmp_path_factory) -> Path:
    """An fp16-fix file of the base weights with pre-gate scales (the o / ffn_down rows
    folded too), written at the base's own dtype."""
    out = tmp_path_factory.mktemp("built_fix") / "fix.safetensors"
    base = weights_root / "base"
    sets = {
        "WEIGHTS": str(base),
        **BUILT_FIX,
        "SAVE_DTYPE": DTYPE_NAMES[stored_dtype(base)],
        "OUT": str(out),
    }
    run_script(str(FIX_SCRIPT), set_overrides=sets)
    return out


def unfolded(path) -> dict:
    """The backbone weights of `path` (fp32, CPU) with the recorded fold undone by the
    model's own inverse (every parameter it scales: weight rows, biases, modulation)."""
    model, _ = load_backbone(str(path), device="cpu", dtype=torch.float32)
    unfold(model)
    return {k: v.detach().float() for k, v in model.state_dict().items()}


def check_same_delta(d_fix: dict, d_plain: dict, slack: dict) -> None:
    """Per tensor: the fix-file merge's delta W (unfolded) equals the plain merge's,
    within 1e-5 x its max plus the per-element `slack` (the fp32 rounding of the merged
    weights and each base dtype's rounding of delta W)."""
    assert d_fix.keys() == d_plain.keys()
    for key, d_a in d_plain.items():
        bound = 1e-5 * float(d_a.abs().max()) + slack[key]
        assert bool(((d_fix[key] - d_a).abs() <= bound).all()), key


def delta_rounding(*paths: Path) -> tuple[float, float]:
    """(relative, absolute) rounding of a delta W computed in the coarser of the
    checkpoints' stored dtypes: half an eps, and half an eps of its smallest normal."""
    infos = [torch.finfo(stored_dtype(p)) for p in paths]
    return max(i.eps for i in infos) / 2, max(i.eps * i.tiny for i in infos) / 2


@pytest.mark.parametrize("which", ["built", "shipped"])
def test_merge_into_an_fp16_fix_file(
    which, request, trained, weights_root, fix_file, plain_merge, tmp_path
):
    if which == "shipped" and fix_file is None:
        pytest.skip("no fp16-fix file")
    fix_path = request.getfixturevalue("built_fix") if which == "built" else fix_file
    out = tmp_path / "merged_fix.safetensors"
    _merge(fix_path, _lycoris(trained), out)
    fix_model, _ = load_backbone(str(fix_path), device="cpu", dtype=torch.float32)
    scales = {k: v.float() for k, v in output_row_scales(fix_model).items()}
    params = fold_rows(fix_model)
    del fix_model
    assert any(
        bool((s != 1).any()) for s in scales.values()
    ), "the fix file folds nothing"
    rel, tiny = delta_rounding(weights_root / "base", fix_path)
    base, fix = unfolded(weights_root / "base"), unfolded(fix_path)
    merged, merged_fix = unfolded(plain_merge), unfolded(out)
    out.unlink()
    d_plain = {k: merged[k] - base[k] for k in base}
    d_fix = {k: merged_fix[k] - fix[k] for k in fix}
    slack = {
        k: 2 * FP32_EPS * (merged[k].abs() + merged_fix[k].abs())
        + rel * (d_plain[k].abs() + d_fix[k].abs())
        + tiny
        for k in d_plain
    }
    check_same_delta(d_fix, d_plain, slack)
    bias = next(
        k
        for k, s in params.items()
        if k.endswith(".bias") and isinstance(s, float) and s != 1
    )
    left = {bias: merged_fix[bias] - fix[bias] * float(params[bias])}
    must_fail(
        lambda: check_same_delta(left, {bias: d_plain[bias]}, {bias: slack[bias]}),
        f"the fix file's {bias} left folded",
    )
    folded = {}
    for key, d in d_fix.items():
        s = scales.get(key[: -len(".weight")]) if key.endswith(".weight") else None
        if s is not None and bool((s != 1).any()) and bool(d_plain[key].any()):
            folded[key] = d / s.view(-1, *([1] * (d.dim() - 1)))
    if which == "built":
        assert folded, "the trained adapter touches no row the fold scales"
    for key, dropped in folded.items():
        must_fail(
            lambda: check_same_delta(
                {key: dropped}, {key: d_plain[key]}, {key: slack[key]}
            ),
            f"a merge into the fix file that drops the row scales of {key}",
        )


def test_run_writes_its_log(trained):
    text = (trained / "log.txt").read_text(encoding="utf-8")
    assert "training from step" in text, text[-400:]
