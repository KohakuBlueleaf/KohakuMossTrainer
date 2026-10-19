"""Component folders of a weights root: the VARIANT folder's own component, base/'s for
a component the variant does not ship weights of (logged; a config-only folder ships
none), the pipeline dir itself without a variant folder; a missing VARIANT folder is an
error, never a silent base/ fallback."""

import logging

import pytest

from kohakumoss.inference.loading import component_root
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def tree(root, folders: list[str], configs: tuple[str, ...] = ()):
    """`folders` each with a weights file, `configs` each with a config.json only."""
    for folder in folders:
        (root / folder).mkdir(parents=True)
        (root / folder / "model.safetensors").write_bytes(b"")
    for folder in configs:
        (root / folder).mkdir(parents=True)
        (root / folder / "config.json").write_text("{}")
    return root


def _refuses_a_missing_variant(choose, root) -> None:
    try:
        choose(root, "sft", "transformer")
    except FileNotFoundError as err:
        assert "VARIANT" in str(err), err
        return
    raise AssertionError("a missing VARIANT folder was accepted")


def _takes_base_weights(choose, root) -> None:
    assert choose(root, "sft", "text_encoder") == root / "base"


def test_variant_folders(tmp_path, caplog):
    root = tree(
        tmp_path / "w",
        ["base/transformer", "base/text_encoder", "base/vae", "sft/transformer"],
        ("sft/text_encoder",),
    )
    assert component_root(root, "sft", "transformer") == root / "sft"
    with caplog.at_level(logging.INFO, logger="kohakumoss"):
        assert component_root(root, "sft", "vae") == root / "base"
    assert "vae/ from" in caplog.text
    _takes_base_weights(component_root, root)
    pipe = tree(tmp_path / "p", ["transformer", "vae"])
    assert component_root(pipe, "base", "vae") == pipe
    only_base = tree(tmp_path / "b", ["base/transformer"])
    _refuses_a_missing_variant(component_root, only_base)

    def falls_back(path, variant, name):
        for folder in (path / variant, path, path / "base"):
            if (folder / name).is_dir():
                return folder
        raise FileNotFoundError(name)

    must_fail(
        lambda: _refuses_a_missing_variant(falls_back, only_base),
        "a silent base/ fallback",
    )
    must_fail(
        lambda: _takes_base_weights(falls_back, root), "a config-only folder taken"
    )
