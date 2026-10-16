"""Application configuration (data roots, cache sizes), from CLI arguments or environment."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Settings for ``waveplot serve``. Environment variables use the ``WAVEPLOT_`` prefix."""

    model_config = SettingsConfigDict(env_prefix="WAVEPLOT_", extra="ignore")

    data_roots: dict[str, Path] = Field(default_factory=dict)
    initial_dataset: str | None = None
    cache_mb: int = 512
    max_open_datasets: int = 8
    title: str = "WavePlot"
    allow_save_next_to_data: bool = True

    @classmethod
    def from_cli(cls, paths: list[Path], data_root: list[str]) -> AppConfig:
        cfg = cls()
        roots: dict[str, Path] = dict(cfg.data_roots)
        initial: str | None = None
        for spec in data_root:
            name, sep, p = spec.partition("=")
            if not sep:
                p, name = name, Path(name).name
            roots[_unique(roots, _clean(name))] = Path(p).expanduser().resolve()
        for p in paths:
            p = p.expanduser().resolve()
            if p.is_file() and p.suffix.lower() == ".map":
                name = _unique(roots, _clean(p.parent.name))
                roots[name] = p.parent
                initial = initial or f"{name}/{p.stem}"
            elif p.is_dir():
                roots[_unique(roots, _clean(p.name))] = p
            else:
                raise FileNotFoundError(p)
        if not roots:
            roots["cwd"] = Path.cwd()
        return cfg.model_copy(update={"data_roots": roots, "initial_dataset": initial or cfg.initial_dataset})


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "data"


def _unique(existing: dict[str, Path], name: str) -> str:
    base, n = name, 2
    while name in existing:
        name = f"{base}{n}"
        n += 1
    return name
