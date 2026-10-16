"""User preferences persisted as JSON in the user's config directory (and in the browser)."""

from __future__ import annotations

import json
from pathlib import Path

from platformdirs import user_config_dir
from pydantic import BaseModel, Field

from ..io.legacy_settings import LegacySettings


class Preferences(BaseModel):
    mode: str = "standard"
    scheme: str = "light"
    n_levels: int = 11
    level_method: str = "lin"
    ess_theta: float = 0.0
    ess_friction: float = 30.0
    fourier: dict[str, float] = Field(default_factory=lambda: {"oversize": 1, "low_freq": 10, "high_freq": 100,
                                                                "low_taper": 5, "high_taper": 95, "max_taper": 5})
    titles: dict[str, str] = Field(default_factory=dict)
    representation: str = "map"


def prefs_path() -> Path:
    return Path(user_config_dir("waveplot", appauthor=False)) / "preferences.json"


def load_preferences() -> Preferences:
    p = prefs_path()
    try:
        return Preferences.model_validate_json(p.read_text())
    except (OSError, ValueError):
        return Preferences()


def save_preferences(prefs: Preferences) -> Path:
    p = prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prefs.model_dump(), indent=2))
    return p


def preferences_from_legacy(s: LegacySettings, base: Preferences | None = None) -> Preferences:
    base = base or load_preferences()
    return base.model_copy(update={
        "n_levels": max(2, min(int(s.qty_levels or 11), 256)),
        "level_method": s.level_method,
        "ess_theta": float(s.theta),
        "ess_friction": float(s.friction) if s.friction else base.ess_friction,
        "fourier": {"oversize": s.freq_wind, "low_freq": s.low_freq, "high_freq": s.high_freq,
                    "low_taper": s.low_taper, "high_taper": s.high_taper, "max_taper": s.max_taper},
        "titles": {"main": s.hd_title, "horizontal": s.hz_title, "vertical": s.vt_title},
        "representation": s.representation,
    })
