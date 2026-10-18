"""Component folders of a weights root: the VARIANT folder's own component, base/'s for
a component the variant does not ship (logged), the pipeline dir itself without a
variant folder; a missing VARIANT folder is an error, never a silent base/ fallback."""

import logging

import pytest

from kohakumoss.inference.loading import component_root
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def tree(root, folders: list[str]):
    for folder in folders:
        (root / folder).mkdir(parents=True)
    return root


def _refuses_a_missing_variant(choose, root) -> None:
    try:
        choose(root, "sft", "transformer")
    except FileNotFoundError as err:
        assert "VARIANT" in str(err), err
        return
    raise AssertionError("a missing VARIANT folder was accepted")


def test_variant_folders(tmp_path, caplog):
    root = tree(
        tmp_path / "w",
        ["base/transformer", "base/text_encoder", "base/vae", "sft/transformer"],
    )
    assert component_root(root, "sft", "transformer") == root / "sft"
    with caplog.at_level(logging.INFO, logger="kohakumoss"):
        assert component_root(root, "sft", "vae") == root / "base"
    assert "vae/ from" in caplog.text
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
