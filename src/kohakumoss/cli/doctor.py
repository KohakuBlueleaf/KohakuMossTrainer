"""Environment report: OS, Python, torch, GPUs and their arch, toolchain, kernel matrix,
and what the install lacks for this machine with the command that fixes it.

    kohakumoss-doctor [--json report.json] [--memory]

The report holds no paths, host or user names.
"""

import argparse
import importlib.metadata as metadata
import json
import platform
import re
import shutil
import sys
from pathlib import Path

import torch

from kohakumoss.cli.missing import missing
from kohakumoss.logging.log import render
from kohakumoss.memory.profiles import ORDER
from kohakumoss.memory.profiles import resolve as resolve_profile
from kohakumoss_common.units import gib
from kohakumoss_kernels import available, registry, status
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.hardware.toolchain import compilers, msvc_compilers, vswhere

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
PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\",;]*|(?<![\w.])/[\w.-]+/[^\s'\",;]*")


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
                    "memory_gib": round(gib(props.total_memory), 2),
                    "bf16": arch.has_bf16,
                    "fp8": arch.has_fp8,
                }
            )
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            out.append(
                {
                    "index": i,
                    "name": props.name,
                    "arch": "xpu",
                    "memory_gib": round(gib(props.total_memory), 2),
                }
            )
    return out


def device_kinds() -> list[str]:
    """One device per accelerator kind present, then cpu (the kernel matrix rows)."""
    kinds = [k for k in ("cuda", "xpu") if getattr(torch, k).is_available()]
    return [f"{k}:0" for k in kinds] + ["cpu"]


def msvc() -> dict:
    """MSVC presence on Windows (vswhere, then a direct search; nvcc needs cl.exe)."""
    if sys.platform != "win32":
        return {"needed": False}
    found = shutil.which("cl") is not None
    via_vswhere = bool(vswhere("-latest", "-property", "installationPath"))
    versions_found = sorted({Path(p).parts[-5] for p in msvc_compilers()})
    return {
        "needed": True,
        "cl_on_path": found,
        "vswhere_sees_vs": via_vswhere,
        "msvc_versions": versions_found,
    }


def _no_paths(text: str) -> str:
    return PATH_RE.sub("<path>", text)


def backend_failures() -> dict[str, str]:
    """Backends that failed an op here, this process and earlier runs (paths removed)."""
    out = {k: v["error"] for k, v in registry.recorded_failures().items()}
    out.update(registry.failures())
    return {k: _no_paths(v)[:300] for k, v in sorted(out.items())}


def kernel_matrix(device: str) -> dict:
    out = {}
    for dtype in (torch.float16, torch.bfloat16):
        out[str(dtype).split(".")[-1]] = {
            op: {b: (r is None) for b, r in available(op, device, dtype).items()}
            for op in kernel_ops.ALL
        }
    return out


def memory_plans(device: str) -> dict:
    """Each named memory profile resolved for `device`."""
    out = {}
    for profile in ORDER:
        try:
            plan = resolve_profile(profile, device=device)
            out[profile] = {k: str(v) for k, v in vars(plan).items()}
        except (ValueError, RuntimeError) as err:
            out[profile] = {"error": str(err)}
    return out


def report(memory: bool = False) -> dict:
    kinds = device_kinds()
    extra = {"memory_plans": memory_plans(kinds[0])} if memory else {}
    kernels = {dev: kernel_matrix(dev) for dev in kinds}
    toolchain = msvc()
    smoke = {
        k: None if v is None else _no_paths(v)[:300]
        for k, v in status.status_report().items()
    }
    return {
        **extra,
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "packages": versions(),
        "devices": devices(),
        "missing": missing(toolchain, smoke),
        "msvc": toolchain,
        "compilers": {k: _no_paths(v) for k, v in compilers().items()},
        "kernels": kernels,
        "backend_status": smoke,
        "backend_failures": backend_failures(),
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
    for backend, compiler in rep["compilers"].items():
        print(f"  compiler {backend:15s} {compiler}")
    for key, why in rep["backend_status"].items():
        print(f"  status {key:20s} {'ok' if why is None else why}")
    for key, error in rep["backend_failures"].items():
        print(f"  FAILED {key}: {error}")
    for dev, tables in rep["kernels"].items():
        for dtype, table in tables.items():
            for op, row in table.items():
                ok = ", ".join(b for b, v in row.items() if v) or "none"
                print(f"  {dev:7s} {dtype:8s} {op:24s} {ok}")
    _print_missing(rep["missing"])


def _print_missing(rows: list[dict]) -> None:
    if not rows:
        print("missing: nothing")
        return
    print("missing:")
    for row in rows:
        print(f"  - {row['what']}\n      fix: {row['fix']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KohakuMoss environment report")
    parser.add_argument("--json", default="", help="also write the report as JSON")
    parser.add_argument(
        "--memory", action="store_true", help="resolve every memory profile"
    )
    parser.add_argument(
        "--clear-failures",
        action="store_true",
        help="forget recorded backend failures before reporting",
    )
    args = parser.parse_args()
    if args.clear_failures:
        status.clear_failures()
    rep = report(args.memory)
    _print(rep)
    for profile, plan in rep.get("memory_plans", {}).items():
        print(f"  profile {profile}:")
        print("\n".join(render(plan, depth=2)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
