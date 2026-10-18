"""The install helpers read the driver's CUDA version from both nvidia-smi header
spellings: "CUDA Version: X.Y" and, on newer drivers, "CUDA UMD Version: X.Y"
(install.sh's sed expression, install.ps1's regex, scripts/colab/setup.py)."""

import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
HEADERS = {
    "| NVIDIA-SMI 595.99                 Driver Version: 595.99         "
    "CUDA Version: 13.2     |": (13, 2),
    "| NVIDIA-SMI 615.78.08              KMD Version: 615.78.08     "
    "CUDA UMD Version: 13.4 |": (13, 4),
    "| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      "
    "CUDA Version: 12.6     |": (12, 6),
}
NO_CUDA = "| NVIDIA-SMI 615.78.08              KMD Version: 615.78.08 |"
OLD_PATTERN = r"CUDA Version:\s*(\d+)\.(\d+)"


def check(parse) -> None:
    """`parse(nvidia-smi text)` gives (major, minor) for every header, None without one."""
    for header, want in HEADERS.items():
        assert parse(f"{header}\n|----|\n") == want, header
    assert parse(NO_CUDA) is None


def _regex_parse(pattern: str):
    def parse(text: str):
        found = re.search(pattern, text)
        return None if found is None else (int(found[1]), int(found[2]))

    return parse


def _sed_parse(expr: str):
    def parse(text: str):
        out = subprocess.run(
            ["sed", "-n", expr], input=text, capture_output=True, text=True, check=True
        ).stdout.split()
        return (int(out[0]), int(out[1])) if out else None

    return parse


def test_colab_setup_reads_both_spellings():
    setup = runpy.run_path(
        str(ROOT / "scripts/colab/setup.py"), run_name="__import_check__"
    )
    check(setup["parse_cuda"])
    must_fail(lambda: check(_regex_parse(OLD_PATTERN)), "the old spelling only")


def test_install_ps1_regex_reads_both_spellings():
    text = (ROOT / "scripts/install/install.ps1").read_text(encoding="utf-8")
    found = re.search(
        r'\[regex\]::Match\(\(nvidia-smi \| Out-String\), "([^"]+)"\)', text
    )
    assert found is not None, "install.ps1 reads no CUDA version"
    check(_regex_parse(found[1]))


@pytest.mark.skipif(shutil.which("sed") is None, reason="no sed on this host")
def test_install_sh_sed_reads_both_spellings():
    text = (ROOT / "scripts/install/install.sh").read_text(encoding="utf-8")
    found = re.search(r"nvidia-smi \| sed -n '([^']+)'", text)
    assert found is not None, "install.sh reads no CUDA version"
    check(_sed_parse(found[1]))
    old = r"s/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p"
    must_fail(lambda: check(_sed_parse(old)), "the old sed spelling only")
