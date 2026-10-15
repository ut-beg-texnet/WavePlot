"""Receivers: histories recorded at the same grid point, with component triplets detected."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..io.records import HistRecord

VELOCITY_SETS = (("XVEL", "YVEL", "ZVEL"), ("aXVEL", "aYVEL", "aZVEL"))
DISPLACEMENT_SET = ("Xdisp", "Ydisp", "Zdisp")
STRESS_SET = ("S11", "S22", "S33", "S12", "S23", "S31")


@dataclass
class Receiver:
    number: int
    position: tuple[int, int, int]
    records: list[HistRecord] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.records]

    def by_name(self, name: str) -> HistRecord | None:
        for r in self.records:
            if r.name.lower() == name.lower():
                return r
        return None

    @property
    def velocity(self) -> tuple[HistRecord | None, HistRecord | None, HistRecord | None] | None:
        for names in VELOCITY_SETS:
            recs = tuple(self.by_name(n) for n in names)
            if sum(r is not None for r in recs) >= 2:
                return recs  # type: ignore[return-value]
        return None

    @property
    def stress(self) -> dict[str, HistRecord]:
        return {n: r for n in STRESS_SET if (r := self.by_name(n)) is not None}

    @property
    def label(self) -> str:
        i, j, k = self.position
        return f"R{self.number} ({i},{j},{k}): {', '.join(self.names)}"


def build_receivers(hists: list[HistRecord]) -> list[Receiver]:
    groups: dict[tuple[int, int, int], list[HistRecord]] = {}
    for h in hists:
        if h.is_global or h.is_profile:
            continue
        groups.setdefault(h.position, []).append(h)
    return [Receiver(n, pos, recs) for n, (pos, recs) in enumerate(groups.items(), start=1)]
