"""Set up KohakuMoss on a Google Colab VM: a uv venv with the torch build matching the
driver's CUDA, the package extras, an optional editable LyCORIS checkout, then the
doctor report. Runs with the VM's system Python; only uv is installed into it.
See docs/guides/install.md (Google Colab).

    python scripts/colab/setup.py
    python scripts/colab/setup.py --venv /content/km-venv --lycoris /content/LyCORIS
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
TORCH_INDEX = ((13, 0, "cu130"), (12, 8, "cu128"), (12, 6, "cu126"))
CUDA_RE = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")
BUILD_TOOLS = ("setuptools", "wheel", "ninja", "packaging")


def run(cmd: list[str], check: bool = True) -> int:
    """Echo and run `cmd`; returns its exit code."""
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check).returncode


def parse_cuda(text: str) -> tuple[int, int] | None:
    """(major, minor) of nvidia-smi's "CUDA Version: X.Y" (or, on newer drivers, "CUDA
    UMD Version: X.Y") header field; None when `text` has neither."""
    found = CUDA_RE.search(text)
    return None if found is None else (int(found[1]), int(found[2]))


def driver_cuda() -> tuple[int, int]:
    """Highest CUDA version the installed driver supports, from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=True
    ).stdout
    found = parse_cuda(out)
    if found is None:
        raise SystemExit(
            "nvidia-smi reports no CUDA version: is a GPU runtime attached?"
        )
    return found


def torch_index(cuda: tuple[int, int]) -> str:
    """PyTorch wheel index of the newest CUDA build the driver runs."""
    for major, minor, tag in TORCH_INDEX:
        if cuda >= (major, minor):
            return f"https://download.pytorch.org/whl/{tag}"
    raise SystemExit(
        f"driver CUDA {cuda[0]}.{cuda[1]} < 12.6: no torch wheel runs on it"
    )


def uv_bin() -> str:
    """Path of uv; installs it into the system Python's user site when missing."""
    found = shutil.which("uv")
    if found:
        return found
    run([sys.executable, "-m", "pip", "install", "-q", "--user", "uv"])
    return str(Path.home() / ".local" / "bin" / "uv")


def project_table() -> dict:
    """The `[project]` table of pyproject.toml."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def torch_requirement() -> str:
    """The torch requirement pyproject.toml pins."""
    deps = project_table()["dependencies"]
    return next(d for d in deps if re.match(r"torch\s*[<>=!~]", d))


def extra_requirements(extras: list[str]) -> list[str]:
    """Requirement strings of `extras` as pyproject.toml declares them."""
    table = project_table()["optional-dependencies"]
    unknown = sorted(set(extras) - set(table))
    if unknown:
        raise SystemExit(f"unknown extras {unknown}; pyproject has {sorted(table)}")
    return [req for name in extras for req in table[name]]


def install(uv: str, py: str, args) -> list[str]:
    """torch, LyCORIS, KohakuMoss + extras into the venv; returns requirements that
    failed to install (the base package never fails silently)."""
    index = args.torch_index or torch_index(driver_cuda())
    pip = [uv, "pip", "install", "--python", py]
    run(pip + [args.torch, "torchvision", "--index-url", index])
    if args.lycoris:
        run(pip + ["-e", args.lycoris])
    extras = [e for e in args.extras.split(",") if e]
    spec = f"{ROOT}[{','.join(extras)}]" if extras else str(ROOT)
    if run(pip + ["-e", spec], check=False) == 0:
        return []
    run(pip + ["-e", str(ROOT)])
    run(pip + list(BUILD_TOOLS))
    failed = []
    for req in extra_requirements(extras):
        if run(pip + ["--no-build-isolation", req], check=False) != 0:
            failed.append(req)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default="/content/km-venv")
    ap.add_argument(
        "--python",
        default=f"{sys.version_info.major}.{sys.version_info.minor}",
        help="venv Python version (default: the VM's system Python)",
    )
    ap.add_argument(
        "--torch", default=torch_requirement(), help="torch requirement (pyproject's)"
    )
    ap.add_argument("--torch-index", default="", help="wheel index; '' = by driver")
    ap.add_argument("--extras", default="kernels,lightning,quant,dev,bench")
    ap.add_argument("--lycoris", default="", help="LyCORIS checkout to install -e")
    ap.add_argument("--doctor-json", default="", help="also write the doctor report")
    args = ap.parse_args()

    uv = uv_bin()
    venv = Path(args.venv)
    run([uv, "venv", str(venv), "--python", args.python, "--allow-existing"])
    py = str(venv / "bin" / "python")
    failed = install(uv, py, args)

    doctor = [str(venv / "bin" / "kohakumoss-doctor"), "--memory"]
    if args.doctor_json:
        doctor += ["--json", args.doctor_json]
    run(doctor)
    for req in failed:
        print(f"optional requirement not installed: {req}")
    print(f"\nvenv ready: {venv}\n  source {venv}/bin/activate")


if __name__ == "__main__":
    main()
