"""Locating WAVE datasets on disk.

A dataset is a family of files sharing a base name: ``<name>.map`` (required) plus optional
``.snp .hst .dmp .geo .crk .otl .m .log``. Extensions are matched case-insensitively so DOS-era
upper-case files work on Linux shares too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SIDE_EXTENSIONS = ("map", "snp", "hst", "dmp", "geo", "crk", "otl", "m", "log", "psc")


@dataclass(frozen=True)
class DatasetFiles:
    stem: str
    folder: Path
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def map(self) -> Path:
        return self.files["map"]

    def get(self, ext: str) -> Path | None:
        return self.files.get(ext.lower())


def resolve_dataset(path: str | os.PathLike[str]) -> DatasetFiles:
    """Resolve a ``.map`` path, a base path without extension, or a folder with one map."""
    p = Path(path).expanduser()
    if p.is_dir():
        maps = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".map")
        if len(maps) != 1:
            raise FileNotFoundError(f"{p}: expected exactly one .map file, found {len(maps)}")
        p = maps[0]
    if p.suffix.lower() != ".map":
        candidates = [c for c in (p.with_name(p.name + ".map"), p.with_name(p.name + ".MAP")) if c.is_file()]
        if not candidates and p.parent.is_dir():
            candidates = [f for f in p.parent.iterdir()
                          if f.is_file() and f.suffix.lower() == ".map" and f.stem.lower() == p.name.lower()]
        if not candidates:
            raise FileNotFoundError(f"no .map file for {p}")
        p = candidates[0]
    if not p.is_file():
        raise FileNotFoundError(p)
    folder, stem = p.parent, p.stem
    files: dict[str, Path] = {}
    for f in folder.iterdir():
        if f.is_file() and f.stem.lower() == stem.lower():
            ext = f.suffix.lower().lstrip(".")
            if ext in SIDE_EXTENSIONS and (ext not in files or f.stem == stem):
                files[ext] = f
    files["map"] = p
    return DatasetFiles(stem=stem, folder=folder, files=files)


def find_maps(root: str | os.PathLike[str], max_depth: int = 4) -> list[Path]:
    """All ``*.map`` files below ``root`` (case-insensitive), skipping hidden folders and venvs."""
    root = Path(root)
    out: list[Path] = []
    skip = {"__pycache__", "node_modules", ".git"}
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base_depth
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in skip and not d.endswith("_venv")
                       and depth < max_depth]
        for fn in filenames:
            if fn.lower().endswith(".map"):
                out.append(Path(dirpath) / fn)
    return sorted(out)
