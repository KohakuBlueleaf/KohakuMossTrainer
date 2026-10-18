"""Config test helpers: the shipped config files."""

from pathlib import Path

from kohakumoss.cli.config.validate import USER_DIR


def shipped_configs(root: Path) -> list[Path]:
    """Every config file under configs/ except the written ones (configs/user/)."""
    user = root / USER_DIR
    return sorted(
        p
        for p in (root / "configs").rglob("*.py")
        if user not in p.parents and "__pycache__" not in p.parts
    )
