"""Resolve the SQLite inventory independently from the external payload volume."""
import os
from pathlib import Path


def inventory_path(packed_root: Path, explicit=None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get("HK3D_DB_PATH"):
        return Path(os.environ["HK3D_DB_PATH"]).expanduser()
    pointer = packed_root / "inventory.pointer"
    if pointer.exists():
        return Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
    return packed_root / "inventory.sqlite"


def write_pointer(packed_root: Path, database: Path) -> None:
    default = packed_root / "inventory.sqlite"
    if database.absolute() != default.absolute():
        (packed_root / "inventory.pointer").write_text(str(database.absolute()) + "\n", encoding="utf-8")
