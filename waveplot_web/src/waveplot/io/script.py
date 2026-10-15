"""Light-weight parsers for WAVE text side files: input script (``.m``), run log (``.log``)
and stope outline (``.otl``). They provide metadata only; the ``.map`` file stays authoritative.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
_KV = re.compile(r"(\w+)\s*=\s*([^\s]+)")


def _f(s: str) -> float:
    return float(s.replace("d", "e").replace("D", "e"))


@dataclass
class ScriptProp:
    material: int
    values: dict[str, float]

    @property
    def vp(self) -> float | None:
        v = self.values
        if "cp" in v:
            return v["cp"]
        if {"b", "s", "dens"} <= v.keys() and v["dens"]:
            return math.sqrt((v["b"] + 4 * v["s"] / 3) / v["dens"])
        return None

    @property
    def vs(self) -> float | None:
        v = self.values
        if "cs" in v:
            return v["cs"]
        if {"s", "dens"} <= v.keys() and v["dens"]:
            return math.sqrt(v["s"] / v["dens"])
        return None


@dataclass
class ScriptHistory:
    var: str
    position: tuple[int, int, int] | None


@dataclass
class WaveScript:
    path: str
    grid: tuple[int, int, int] | None = None
    spacing: tuple[float, float, float] | None = None
    props: list[ScriptProp] = field(default_factory=list)
    materials: list[tuple[int, dict[str, str]]] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    stopes: list[dict[str, str]] = field(default_factory=list)
    histories: list[ScriptHistory] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    cycles: list[int] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)


_HIS_ITEM = re.compile(r"([A-Za-z_][\w-]*)\s*(?:\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\))?")


def parse_script(path: str | Path) -> WaveScript:
    path = Path(path)
    sc = WaveScript(path=str(path))
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        sc.commands.append(line)
        words = line.split()
        cmd = words[0].lower()
        if cmd == "gr" and len(words) >= 4:
            try:
                sc.grid = (int(words[1]), int(words[2]), int(words[3]))
                if len(words) >= 7:
                    sc.spacing = (_f(words[4]), _f(words[5]), _f(words[6]))
            except ValueError:
                pass
        elif cmd == "prop":
            kv = dict(_KV.findall(line))
            try:
                mat = int(kv.pop("mat", "1"))
                sc.props.append(ScriptProp(mat, {k.lower(): _f(v) for k, v in kv.items()}))
            except ValueError:
                pass
        elif cmd == "mat" and len(words) > 1:
            try:
                sc.materials.append((int(words[1]), dict(_KV.findall(line))))
            except ValueError:
                pass
        elif cmd == "so":
            sc.sources.append(dict(_KV.findall(line)))
        elif cmd == "stope":
            sc.stopes.append(dict(_KV.findall(line)))
        elif cmd == "his":
            body = line[3:]
            for m in _HIS_ITEM.finditer(body):
                name = m.group(1)
                if "=" in body[m.end():m.end() + 1] or name.lower() in ("dt", "all"):
                    continue
                pos = tuple(int(g) for g in m.group(2, 3, 4)) if m.group(2) else None
                sc.histories.append(ScriptHistory(name, pos))  # type: ignore[arg-type]
        elif cmd == "pl":
            sc.plots.append(line)
        elif cmd in ("cy", "cyc", "cycle") and len(words) > 1:
            try:
                sc.cycles.append(int(words[1]))
            except ValueError:
                pass
    return sc


@dataclass
class RunLog:
    path: str
    started: str | None = None
    stopped: str | None = None
    timesteps: list[float] = field(default_factory=list)
    cycles: list[tuple[int, float]] = field(default_factory=list)   # (cycle, time)
    messages: list[str] = field(default_factory=list)

    @property
    def timestep(self) -> float | None:
        return self.timesteps[-1] if self.timesteps else None

    @property
    def final_cycle(self) -> tuple[int, float] | None:
        return self.cycles[-1] if self.cycles else None


_TS = re.compile(r"timestep\s*=\s*(" + _NUM + ")", re.I)
_CY = re.compile(r"Cycle\s+(\d+)\s*,\s*time\s*=\s*(" + _NUM + ")", re.I)


def parse_log(path: str | Path) -> RunLog:
    path = Path(path)
    log = RunLog(path=str(path))
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip()
        if (m := _TS.search(line)):
            log.timesteps.append(_f(m.group(1)))
        elif (m := _CY.search(line)):
            log.cycles.append((int(m.group(1)), _f(m.group(2))))
        elif "log-file" in line and "opened" in line:
            log.started = line.split("opened", 1)[1].strip() or None
        elif "recording stopped" in line:
            log.stopped = line.split("stopped", 1)[1].strip() or None
        elif line.strip().startswith("**") or line.strip().startswith("* "):
            log.messages.append(line.strip())
    return log


def parse_outline(path: str | Path) -> list[tuple[int, int]]:
    """``.otl``: first line = number of points, then one ``x y`` integer pair per line."""
    lines = [ln.split() for ln in Path(path).read_text(errors="replace").splitlines() if ln.strip()]
    if not lines:
        return []
    tot = int(lines[0][0])
    pts = []
    for parts in lines[1:tot + 1]:
        if len(parts) >= 2:
            pts.append((int(parts[0]), int(parts[1])))
    return pts
