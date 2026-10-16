"""Environment report: OS, Python, torch, GPUs and their arch, toolchain, kernel matrix.

    kohakumoss-doctor [--json report.json] [--memory]

The report holds no paths, host or user names.
"""

import argparse
import glob
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys

import torch

from kohakumoss.training.build.integration import MEMORY_RESOLVE
from kohakumoss.training.build.model import optional_object
from kohakumoss_kernels import available
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels.hardware.arch import arch_of

PROFILE_NAMES = ("speed", "balanced", "low", "minimal")

PACKAGES = (
    "torch",
    "triton",
    "triton-windows",
    "tilelang",
    "lycoris-lora",
    "kohaku-engine",
    "anyschedule",
    "kohakuvault",
    "lightning",
    "diffusers",
    "transformers",
    "bitsandbytes",
)
VSWHERE = r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
MSVC_GLOB = (
    r"%ProgramFiles%\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
)


def versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def devices() -> list[dict]:
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            arch = arch_of(f"cuda:{i}")
            out.append(
                {
                    "index": i,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "arch": arch.name,
                    "memory_gib": round(props.total_memory / 1024**3, 2),
                    "bf16": arch.has_bf16,
                    "fp8": arch.has_fp8,
                }
            )
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            out.append(
                {"index": i, "name": torch.xpu.get_device_name(i), "arch": "xpu"}
            )
    return out


def msvc() -> dict:
    """MSVC presence on Windows (vswhere, then a direct search; nvcc needs cl.exe)."""
    if sys.platform != "win32":
        return {"needed": False}
    found = shutil.which("cl") is not None
    vswhere = os.path.expandvars(VSWHERE)
    via_vswhere = False
    if os.path.exists(vswhere):
        res = subprocess.run(
            [vswhere, "-latest", "-products", "*", "-property", "installationVersion"],
            capture_output=True,
            text=True,
        )
        via_vswhere = bool(res.stdout.strip())
    direct = sorted(glob.glob(os.path.expandvars(MSVC_GLOB)))
    versions_found = sorted({p.split(os.sep)[-5] for p in direct})
    return {
        "needed": True,
        "cl_on_path": found,
        "vswhere_sees_vs": via_vswhere,
        "msvc_versions": versions_found,
    }


def kernel_matrix(device: str) -> dict:
    out = {}
    for dtype in (torch.float16, torch.bfloat16):
        out[str(dtype).split(".")[-1]] = {
            op: {b: (r is None) for b, r in available(op, device, dtype).items()}
            for op in kernel_ops.ALL
        }
    return out


def memory_plans(device: str) -> dict:
    """Each named memory profile resolved for `device` (empty when unavailable)."""
    resolve = optional_object(MEMORY_RESOLVE)
    if resolve is None:
        return {}
    out = {}
    for profile in PROFILE_NAMES:
        try:
            out[profile] = {
                k: str(v) for k, v in vars(resolve(profile, device=device)).items()
            }
        except (ValueError, RuntimeError) as err:
            out[profile] = {"error": str(err)}
    return out


def report(memory: bool = False) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    extra = {"memory_plans": memory_plans(dev)} if memory else {}
    return {
        **extra,
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "packages": versions(),
        "devices": devices(),
        "msvc": msvc(),
        "kernels": kernel_matrix(dev),
    }


def _print(rep: dict) -> None:
    print(
        f"{rep['os']} {rep['os_release']} | Python {rep['python']} | CUDA {rep['torch_cuda']}"
    )
    for name, ver in rep["packages"].items():
        print(f"  {name:16s} {ver or '-'}")
    for d in rep["devices"]:
        print(
            f"  device {d['index']}: {d['name']} [{d['arch']}] {d.get('memory_gib', '?')} GiB"
        )
    if rep["msvc"]["needed"]:
        print(f"  MSVC: {rep['msvc']}")
    for dtype, table in rep["kernels"].items():
        for op, row in table.items():
            ok = ", ".join(b for b, v in row.items() if v) or "none"
            print(f"  {dtype:8s} {op:24s} {ok}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KohakuMoss environment report")
    parser.add_argument("--json", default="", help="also write the report as JSON")
    parser.add_argument(
        "--memory", action="store_true", help="resolve every memory profile"
    )
    args = parser.parse_args()
    rep = report(args.memory)
    _print(rep)
    for profile, plan in rep.get("memory_plans", {}).items():
        print(f"  profile {profile}: {plan}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
