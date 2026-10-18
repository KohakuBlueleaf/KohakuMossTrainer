"""Build hook: wheels carry configs/ and scripts/ as kohakumoss/bundle/ package data, so a
`pip install kohakumoss` has the presets and scripts without a clone. Editable installs
read them from the checkout. See docs/guides/install.md."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
BUNDLED = ("configs", "scripts")
# user configs and the bench matrix dev package (scripts/bench/matrix) stay out
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "user", "matrix")


def bundle(target: Path, root: Path = ROOT, skip=SKIP) -> None:
    """Copy BUNDLED from `root` into `target` (the wheel's kohakumoss/bundle/)."""
    for name in BUNDLED:
        shutil.copytree(root / name, target / name, dirs_exist_ok=True, ignore=skip)


class BuildPy(build_py):
    """build_py that also copies BUNDLED into <build_lib>/kohakumoss/bundle/."""

    def run(self):
        super().run()
        if getattr(self, "editable_mode", False):
            return
        bundle(Path(self.build_lib) / "kohakumoss" / "bundle")


if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildPy})
