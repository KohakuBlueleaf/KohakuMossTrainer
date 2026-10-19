"""`kohakumoss-config new`: every supported hardware x algorithm x data combination
writes a train config, a data config and a sample config that load back to the composed
knobs and pass the knob checks; unsupported ones are refused before anything is
written."""

import sys
from pathlib import Path

import pytest
import torch
from kohakuengine import load_config_file

from kohakumoss.cli.config import main as cli
from kohakumoss.cli.config import presets, starter
from kohakumoss.cli.config.validate import (
    BASES,
    SAMPLE,
    TRAIN,
    blocking,
    compose,
    config_knobs,
    config_problems,
    load_knobs,
    problems,
)
from kohakumoss_data import DataConfig, resolved_cache_dir
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ALGOS = [a for a in presets.names(ROOT, "moss", "algo") if a != "full"]
SOURCES = presets.names(ROOT, "moss", "data")
VRAMS = ["8", "12", "16", "24", "48", "auto"]


@pytest.fixture(autouse=True)
def linux_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")


def new_train(tmp_path: Path, *flags: str, name: str = "run") -> tuple[Path, Path]:
    for sub in ("w", "d"):
        (tmp_path / sub).mkdir(exist_ok=True)
    out = tmp_path / f"{name}.py"
    cli.main(
        [
            "new",
            "train",
            "--weights",
            str(tmp_path / "w"),
            "--data",
            str(tmp_path / "d"),
        ]
        + ["--name", name, "--out", str(out), *flags]
    )
    return out, tmp_path / f"{name}_data.py"


def assert_written_valid(path: Path) -> None:
    bad = blocking(config_problems(path, ROOT))
    assert bad == [], bad


def assert_ascii(path: Path) -> None:
    path.read_bytes().decode("ascii")


@pytest.mark.parametrize("vram", VRAMS)
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("algo", ADAPTER_ALGOS)
def test_adapter_matrix(tmp_path, algo, precision, vram):
    out, data = new_train(
        tmp_path, "--algo", algo, "--vram", vram, "--precision", precision
    )
    for path in (out, data):
        assert_written_valid(path)
        assert_ascii(path)
    _, k = config_knobs(out, ROOT)
    assert k["ALGO"] == ("lora" if algo == "dora" else algo)
    assert k["PRECISION"] == precision


@pytest.mark.parametrize("source", SOURCES)
def test_sources(tmp_path, source):
    out, data = new_train(
        tmp_path, "--source", source, "--vram", "24", "--precision", "bf16"
    )
    _, k = config_knobs(out, ROOT)
    template = load_config_file(presets.layer(ROOT, "moss", "data", source))
    expected = {
        **template.globals_dict["DATASETS"][0],
        "path": starter.posix(tmp_path / "d"),
    }
    assert k["DATASETS"] == [expected]
    _, kd = config_knobs(data, ROOT)
    assert kd["DATASETS"] == k["DATASETS"]
    assert resolved_cache_dir(DataConfig.from_knobs(kd)) == resolved_cache_dir(
        DataConfig.from_knobs(k)
    )
    assert kd["IMAGE_ENCODER"] == "kohakumoss.encoders.vae_encoder"


@pytest.mark.parametrize("vram", ["24", "32", "auto"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_full_finetune(tmp_path, vram, precision):
    out, _ = new_train(
        tmp_path, "--algo", "full", "--vram", vram, "--precision", precision
    )
    assert_written_valid(out)


@pytest.mark.parametrize("vram", VRAMS)
def test_ddp(tmp_path, vram):
    out, _ = new_train(tmp_path, "--gpus", "2", "--vram", vram, "--precision", "bf16")
    assert_written_valid(out)
    _, k = config_knobs(out, ROOT)
    assert k["PARALLEL"] == "ddp" and k["DEVICES"] == (0, 1)


def _full_ft(tmp_path, gpus: str, vram: str) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    out, _ = new_train(
        tmp_path,
        "--algo",
        "full",
        "--gpus",
        gpus,
        "--vram",
        vram,
        "--precision",
        "fp16",
    )
    assert_written_valid(out)
    return config_knobs(out, ROOT)[1]


def _fsdp_on_8_ranks(k: dict) -> None:
    assert k["PARALLEL"] == "fsdp" and k["DEVICES"] == tuple(range(8)), k["PARALLEL"]


def test_full_ft_picks_fsdp_only_where_its_shards_fit(tmp_path, monkeypatch):
    assert _full_ft(tmp_path / "pcie", "8", "32")["PARALLEL"] == "ddp"
    monkeypatch.setattr(starter, "peer_linked", lambda gpus: True)
    _fsdp_on_8_ranks(_full_ft(tmp_path / "a", "8", "32"))
    k = _full_ft(tmp_path / "b", "2", "24")
    swap = load_knobs(presets.layer(ROOT, "moss", "vram", "24g_full"))
    assert k["PARALLEL"] == "ddp" and k["MEMORY_PROFILE"] == swap["MEMORY_PROFILE"]
    must_fail(lambda: _fsdp_on_8_ranks(k), "FSDP on 2 x 24 GB")
    assert _full_ft(tmp_path / "c", "4", "32")["PARALLEL"] == "ddp"


def test_memory_overrides_merge(tmp_path, monkeypatch):
    """MEMORY_OVERRIDES set by two layers merge in layer order."""
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text('MEMORY_OVERRIDES = {"pinned": False, "act_offload": True}\n')
    second.write_text('MEMORY_OVERRIDES = {"act_offload": False}\n')

    def check(names):
        out = starter.merged([first, second], names).get("MEMORY_OVERRIDES")
        assert out == {"pinned": False, "act_offload": False}, out

    check(starter.MERGED)
    must_fail(lambda: check(()), "no MEMORY_OVERRIDES merge")


def test_ddp_layer_keeps_the_vram_profile(tmp_path):
    """The VRAM layer's catalog profile runs unchanged under the ddp layer."""
    out, _ = new_train(tmp_path, "--gpus", "2", "--vram", "12", "--precision", "bf16")
    _, k = config_knobs(out, ROOT)
    assert k["MEMORY_PROFILE"] == "ckpt-ring4-actoff-optstream-gradoff"
    assert not k.get("MEMORY_OVERRIDES"), k.get("MEMORY_OVERRIDES")


def test_written_equals_composed(tmp_path):
    """The rendered file loads back to exactly the in-memory composition."""
    choice = starter.TrainChoice(
        weights=str(tmp_path / "w"),
        data=str(tmp_path / ("d" + chr(0xE9) + "j" + chr(0xE0))),
        vram=12,
        gpus=2,
        prompts=('a "quoted", prompt',),
    )
    st = starter.train_starter(ROOT, choice)
    assert st.layers[0] == ROOT / BASES[TRAIN]
    out = tmp_path / "sub" / "x.py"
    out.parent.mkdir()
    out.write_text(starter.render(st, out, ["kogine run x"], ROOT), encoding="utf-8")
    assert_ascii(out)
    assert config_knobs(out, ROOT)[1] == compose(ROOT, TRAIN, st.layers, st.own)
    st.own["RUN_NAME"] = "changed"
    must_fail(
        lambda: _assert_equal(
            config_knobs(out, ROOT)[1], compose(ROOT, TRAIN, st.layers, st.own)
        ),
        "stale render",
    )


def _assert_equal(a, b):
    assert a == b


def test_layer_paths(tmp_path):
    base = ROOT / BASES[TRAIN]
    assert starter.from_file(base, ROOT / "configs" / "user", ROOT) == (
        "../moss/_base/base_lora.py"
    )
    assert starter.from_file(base, tmp_path, ROOT) == base.resolve().as_posix()
    assert starter.from_file(tmp_path / "a.py", tmp_path, ROOT) == "a.py"
    bundle = tmp_path / "venv" / "kohakumoss" / "bundle"
    preset = bundle / "configs" / "x.py"
    assert starter.from_file(preset, tmp_path, bundle) == preset.resolve().as_posix()


@pytest.mark.parametrize(
    "vram, full, layer",
    [
        (23, False, "24g"),
        (22, False, "16g"),
        (24, False, "24g"),
        (12, False, "12g"),
        (48, False, "auto"),
        (24, True, "24g_full"),
        (80, True, "auto"),
    ],
)
def test_vram_choice(vram, full, layer):
    assert presets.vram_layer(ROOT, vram, full) == layer


@pytest.mark.parametrize("vram, full", [(4, False), (16, True)])
def test_vram_too_small(vram, full):
    with pytest.raises(SystemExit):
        presets.vram_layer(ROOT, vram, full)


def test_refusals_write_nothing(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="Linux-only"):
        monkeypatch.setattr(sys, "platform", "win32")
        new_train(tmp_path, "--gpus", "2", "--vram", "24", "--precision", "bf16")
    assert list(tmp_path.glob("*.py")) == []
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(SystemExit, match="no algo preset"):
        new_train(tmp_path, "--algo", "lorax", "--vram", "24", "--precision", "bf16")
    out, _ = new_train(tmp_path, "--vram", "24", "--precision", "bf16")
    with pytest.raises(SystemExit, match="exists"):
        new_train(tmp_path, "--vram", "24", "--precision", "bf16")
    with pytest.raises(SystemExit, match="pass --vram"):
        new_train(tmp_path, "--force")


@pytest.mark.parametrize("vram", ["4", "8", "12", "24", "32"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_sample(tmp_path, vram, precision):
    out = tmp_path / "s.py"
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"")
    (tmp_path / "w").mkdir()
    cli.main(
        ["new", "sample", "--weights", str(tmp_path / "w"), "--adapter", f"{lora}:0.8"]
        + ["--vram", vram, "--precision", precision, "--out", str(out)]
    )
    assert_written_valid(out)
    script, k = config_knobs(out, ROOT)
    assert script == SAMPLE and k["PRECISION"] == precision
    layer = presets.sample_vram_layer(ROOT, float(vram))
    train = load_knobs(presets.layer(ROOT, "moss", "vram", layer))
    expected = "" if layer == "auto" else train["MEMORY_PROFILE"]
    assert k["MEMORY_PROFILE"] == expected
    assert not k["TEXT_ENCODE"] and not k["VAE_ENCODE"] and not k["TEXT_DEVICE"]
    assert problems(SAMPLE, {**k, "PRECISION": "fp16", "FP16_FIX": "none"})


def test_check_command(tmp_path, capsys):
    out, _ = new_train(tmp_path, "--vram", "24", "--precision", "bf16")
    assert cli.check_configs(ROOT, [str(out)]) == 0
    broken = tmp_path / "broken.py"
    broken.write_text(out.read_text(encoding="utf-8") + "LRR = 1.0\n", encoding="utf-8")
    assert cli.check_configs(ROOT, [str(broken)]) == 1
    assert "did you mean LR" in capsys.readouterr().out
