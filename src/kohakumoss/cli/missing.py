"""What the install lacks for this machine's accelerator, each with the command that fixes
it. Requirements come from the installed kohakumoss metadata (base, then the default
extra of the accelerator kind: `cuda` or `xpu`). See docs/guides/install.md."""

import glob
import importlib.metadata as metadata
import os
import shutil
import sys

import torch
from packaging.requirements import Requirement

PACKAGE = "kohakumoss"
INDEX = "https://download.pytorch.org/whl/{tag}"
INSTALL = (
    'uv pip install -e "{spec}" "torch>=2.13,<2.15" --extra-index-url '
    + INDEX
    + " --index-strategy unsafe-best-match"
)
HELPER = "or run scripts/install/install.sh (Windows: install.ps1)"
TORCH_FIX = (
    'uv pip install --reinstall-package torch "torch>=2.13,<2.15" --index-url '
    "https://download.pytorch.org/whl/cu130 (cu126 / cu128 / cu130 by the driver's "
    f"CUDA version in nvidia-smi), {HELPER}"
)
MSVC_FIX = (
    "install VS Build Tools 2022 with the C++ workload: winget install "
    "Microsoft.VisualStudio.2022.BuildTools --override "
    '"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive" '
    "(the toolchain imports vcvars64 itself)"
)
SDK_GLOB = r"%ProgramFiles(x86)%\Windows Kits\10\Include\*\um"
SDK_FIX = (
    "add the Windows 11 SDK to the Build Tools: winget install "
    "Microsoft.VisualStudio.2022.BuildTools --override "
    '"--add Microsoft.VisualStudio.Component.Windows11SDK.22621 --passive"'
)
NVCC_FIX = (
    "install the CUDA toolkit matching torch.version.cuda "
    "(Windows: winget install Nvidia.CUDA; Linux: the distro's cuda-toolkit package) "
    "and put its bin/ on PATH or set CUDA_PATH"
)
XPU_FIX = (
    'uv pip install --reinstall-package torch "torch>=2.13,<2.15" --index-url '
    f"https://download.pytorch.org/whl/xpu, {HELPER}"
)
BACKEND_FIXES = (
    (
        ("cl.exe", "msvc", "vcvars", "cl "),
        "install the MSVC build tools: see the MSVC row",
    ),
    (("nvcc",), NVCC_FIX),
    (("windows kits", "sdk"), SDK_FIX),
    (
        ("no module named",),
        f'uv pip install -e ".[cuda]" (from the repo root), {HELPER}',
    ),
)


def accelerator_kind() -> str:
    """The accelerator kind torch can use: "cuda" | "xpu", else "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def index_tag(kind: str) -> str:
    """PyTorch wheel index tag of the installed torch build: cuXYZ | xpu | cpu."""
    if kind == "cuda" and torch.version.cuda:
        return "cu" + torch.version.cuda.replace(".", "")
    return kind


def _applies(req: Requirement, extra: str | None) -> bool:
    if req.marker is None:
        return extra is None
    names_extra = "extra" in str(req.marker)
    if extra is None:
        return not names_extra and req.marker.evaluate()
    return names_extra and req.marker.evaluate({"extra": extra})


def requirements(extra: str | None, seen: frozenset = frozenset()) -> list[Requirement]:
    """Requirements of the base package (`extra=None`) or of one extra, self-references
    (`kohakumoss[a,b]`) expanded."""
    out = []
    for text in metadata.requires(PACKAGE) or []:
        req = Requirement(text)
        if not _applies(req, extra):
            continue
        if req.name == PACKAGE:
            for sub in sorted(set(req.extras) - seen):
                out += requirements(sub, seen | {sub})
            continue
        out.append(req)
    return out


def unmet(req: Requirement) -> str | None:
    """Why `req` is not satisfied by the installed packages; None when it is."""
    try:
        version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        return f"{req.name} is not installed"
    if req.specifier and not req.specifier.contains(version, prereleases=True):
        return f"{req.name} {version} does not satisfy {req.specifier}"
    return None


def _torch_rows() -> list[dict]:
    if shutil.which("nvidia-smi") is None or torch.cuda.is_available():
        return []
    if torch.version.cuda is None:
        what = "an NVIDIA GPU is present but torch is a CPU build"
    else:
        what = (
            f"torch is a CUDA {torch.version.cuda} build but sees no device "
            "(driver older than that CUDA?)"
        )
    return [{"what": what, "fix": TORCH_FIX}]


def _msvc_rows(msvc: dict, kind: str) -> list[dict]:
    if sys.platform != "win32" or kind == "cpu":
        return []
    if (
        msvc.get("cl_on_path")
        or msvc.get("vswhere_sees_vs")
        or msvc.get("msvc_versions")
    ):
        return []
    return [
        {
            "what": "no MSVC C++ build tools (kernel compilation needs cl.exe)",
            "fix": MSVC_FIX,
        }
    ]


def windows_sdk() -> bool:
    """True when a Windows 10/11 SDK include tree is present."""
    if os.environ.get("WindowsSdkDir"):
        return True
    return bool(glob.glob(os.path.expandvars(SDK_GLOB)))


def nvcc() -> bool:
    """True when nvcc is on PATH or under CUDA_PATH / CUDA_HOME."""
    if shutil.which("nvcc"):
        return True
    exe = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    homes = [os.environ.get(v, "") for v in ("CUDA_PATH", "CUDA_HOME")]
    return any(h and os.path.isfile(os.path.join(h, "bin", exe)) for h in homes)


def tilelang_builds(probes: dict[str, str | None]) -> bool:
    """True when a TileLang smoke build succeeded on some device here."""
    return any(k.startswith("tilelang@") and v is None for k, v in probes.items())


def _toolchain_rows(kind: str, probes: dict[str, str | None]) -> list[dict]:
    rows = []
    if kind == "cuda" and not nvcc() and not tilelang_builds(probes):
        rows.append(
            {"what": "no CUDA toolkit nvcc (TileLang JIT needs it)", "fix": NVCC_FIX}
        )
    if kind != "cpu" and sys.platform == "win32" and not windows_sdk():
        rows.append(
            {"what": "no Windows SDK (kernel JIT links against it)", "fix": SDK_FIX}
        )
    return rows


def backend_rows(probes: dict[str, str | None]) -> list[dict]:
    """A row per kernel backend whose smoke build failed here, with the likely fix."""
    rows = []
    for key, why in sorted(probes.items()):
        if why is None:
            continue
        text = why.lower()
        fix = next(
            (f for words, f in BACKEND_FIXES if any(w in text for w in words)), ""
        )
        rows.append(
            {"what": f"backend {key} unavailable: {why[:200]}", "fix": fix or "-"}
        )
    return rows


def missing(msvc: dict, probes: dict | None = None) -> list[dict]:
    """[{"what", "fix"}] for every unmet base / default-extra requirement, a torch build
    that cannot use the GPU, absent toolchain parts, and kernel backends whose smoke
    build failed."""
    kind = accelerator_kind()
    tag = index_tag(kind)
    rows = []
    groups = [(None, ".")] + ([(kind, f".[{kind}]")] if kind != "cpu" else [])
    for extra, spec in groups:
        for req in requirements(extra):
            why = unmet(req)
            if why is not None:
                fix = (
                    INSTALL.format(spec=spec, tag=tag)
                    + f" (from the repo root), {HELPER}"
                )
                rows.append({"what": why, "fix": fix})
    probes = probes or {}
    rows += _torch_rows() + _msvc_rows(msvc, kind) + _toolchain_rows(kind, probes)
    rows += _xpu_rows(kind)
    return rows + backend_rows(probes)


def _xpu_rows(kind: str) -> list[dict]:
    if kind != "xpu":
        return []
    rows = []
    if unmet(Requirement("pytorch-triton-xpu")):
        rows.append({"what": "pytorch-triton-xpu is not installed", "fix": XPU_FIX})
    if unmet(Requirement("triton-windows")) is None:
        rows.append(
            {
                "what": "triton-windows is installed next to the XPU torch",
                "fix": "uv pip uninstall triton-windows",
            }
        )
    return rows
