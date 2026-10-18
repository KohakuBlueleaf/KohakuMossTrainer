"""What a cache key tells apart: the encoder's compute spec, the file behind a stat
fingerprint; the caption variants the text pass covers; the spec guard of the cache
passes; the rebuild command's quoting."""

import os
import re
import shlex
from types import SimpleNamespace

import numpy as np
import pytest

from kohakumoss_common.io import write_json_atomic
from kohakumoss_data.cache.build import caption_set, check_spec
from kohakumoss_data.cache.settings import (
    FIELDS,
    offline_miss,
    powershell_quote,
    settings_of,
    settings_path,
)
from kohakumoss_data.cache.tensor_cache import LatentCache, TextCache
from kohakumoss_data.config import DataConfig, encoder_key_id
from kohakumoss_data.sources.base import file_fingerprint
from kohakumoss_data.types import CatalogRow, PlanItem
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROW = CatalogRow(id=0, locator="a.png", w=64, h=64, extra={"hash": "stat:x:1:2"})
ITEM = PlanItem(0, (64, 64), (0, 0, 64, 64))


def text_key(spec: str) -> str:
    return TextCache(None, "off", encoder_key_id("qwen3.5-9b", spec), "bf16").key("a")


def latent_key(spec: str) -> str:
    cache = LatentCache(None, "off", encoder_key_id("flux2-vae", spec), "bf16")
    return cache.key(ROW, ITEM)


def _differ(a: str, b: str) -> None:
    assert a != b


def test_the_encoder_spec_is_part_of_both_keys():
    assert encoder_key_id("qwen3.5-9b", "") == "qwen3.5-9b"
    _differ(text_key("bf16/compute"), text_key("fp16/compute"))
    _differ(text_key("bf16/compute"), text_key("bf16/int8/g0"))
    _differ(latent_key("bf16"), latent_key("fp16"))
    must_fail(lambda: _differ(text_key("bf16/compute"), text_key("bf16/compute")), "")
    assert "latent_encoder_spec" in FIELDS["latents"]
    assert "text_encoder_spec" in FIELDS["text"]
    assert "max_tokens" not in FIELDS["latents"]


def test_equal_stats_are_two_files(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b" / "a.png"
    b.parent.mkdir()
    for path in (a, b):
        path.write_bytes(b"\x89PNG" + bytes(60))
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    _differ(file_fingerprint(str(a)), file_fingerprint(str(b)))

    def size_and_mtime(path) -> str:
        st = os.stat(path)
        return f"stat:{st.st_size}:{st.st_mtime_ns}"

    must_fail(lambda: _differ(size_and_mtime(a), size_and_mtime(b)), "stat only")


def _dataset(variants: int, drawn: int):
    return SimpleNamespace(
        cfg=SimpleNamespace(caption_variants=variants),
        planner=SimpleNamespace(caption_variants=drawn),
        catalog=SimpleNamespace(live_ids=lambda: np.arange(2)),
        caption=lambda row, variant: f"row{row}-v{variant}",
    )


def _covers_variant_0(captions: list[str]) -> None:
    assert {"", "row0-v0", "row1-v0"} <= set(captions), captions


def test_the_text_pass_covers_the_drawn_variants():
    _covers_variant_0(caption_set(_dataset(0, 1)))
    assert len(caption_set(_dataset(3, 3))) == 1 + 2 * 3

    def by_knob(ds) -> list[str]:
        return [""] + [
            ds.caption(r, v)
            for r in ds.catalog.live_ids().tolist()
            for v in range(ds.cfg.caption_variants)
        ]

    must_fail(lambda: _covers_variant_0(by_knob(_dataset(0, 1))), "range(0) variants")


def test_the_cache_pass_refuses_another_spec():
    encode = SimpleNamespace(encoder_spec="fp16")
    check_spec(encode, "fp16", "LATENT_ENCODER_SPEC")
    check_spec(SimpleNamespace(), "bf16", "LATENT_ENCODER_SPEC")
    with pytest.raises(ValueError, match="LATENT_ENCODER_SPEC='bf16'"):
        check_spec(encode, "bf16", "LATENT_ENCODER_SPEC")


def _says_rebuild(msg: str) -> None:
    assert "written under other key knobs" in msg and "TEXT_ENCODER_SPEC" in msg, msg
    assert "kogine run scripts/data/cache_text.py" in msg, msg


def test_builds_lacking_a_key_knob_say_rebuild(tmp_path):
    cfg = DataConfig(cache_dir=str(tmp_path), text_encoder_spec="bf16/compute")
    run = settings_of(cfg, "text")
    old = {k: v for k, v in run.items() if k != "text_encoder_spec"}
    write_json_atomic(settings_path(str(tmp_path), "text"), [old])
    _says_rebuild(str(offline_miss(cfg, "text", 3, "caption 'a'")))
    write_json_atomic(settings_path(str(tmp_path), "text"), [run])
    current = str(offline_miss(cfg, "text", 3, "caption 'a'"))
    must_fail(lambda: _says_rebuild(current), "a build of this version")


def _quoted(quote) -> None:
    """Every value PowerShell would split or expand comes out single-quoted."""
    for text in ("ASPECT_RANGE=0.25,4.0", "X=a;b", "X=$y", "X=@a", "X=(a)", "X=a b"):
        assert quote(text) == "'" + text + "'", (text, quote(text))
    assert quote("X=`a") == "'X=`a'"
    assert quote("X='a'") == "'X=''a'''"


def test_powershell_quoting():
    assert powershell_quote("TARGET_AREA=147456") == "TARGET_AREA=147456"
    text = "CAPTION={'template': 'tags'}"
    assert powershell_quote(text) == "'CAPTION={''template'': ''tags''}'"
    assert shlex.quote(text) != powershell_quote(text)
    _quoted(powershell_quote)
    comma_safe = re.compile(r"[\w@%+=:,./-]*", re.ASCII)

    def with_commas(value: str) -> str:
        return value if comma_safe.fullmatch(value) else powershell_quote(value)

    must_fail(lambda: _quoted(with_commas), "a comma treated as safe")
