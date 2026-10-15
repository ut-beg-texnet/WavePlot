"""WAVE variable numbers, names, units and staggered-grid positions.

Transcribed from the ``VarNames`` / ``VarPositions`` tables of ``DATA_RSC.PAS`` (lines 14-76).
WAVE writes a single integer per record; values above 65535 also carry a "function" code
(max/min/average/... over time), see :func:`split_func`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Group base offsets (DATA_RSC.PAS:20-23)
GRNAM0, GCNAM0, STNAM0, SCNAM0 = 0, 20, 50, 100
GMXNAM0, SMXNAM0, GACNAM0, SACNAM0, MIXNAM0 = 150, 175, 200, 225, 235
FORM_NAME = 255

FUNC_NAMES: tuple[str, ...] = ("", "MX", "mn", "Avg", "Abs", "AMX")
FUNC_DESCRIPTIONS: dict[str, str] = {
    "": "instantaneous value",
    "MX": "maximum over time",
    "mn": "minimum over time",
    "Avg": "average over time",
    "Abs": "absolute value",
    "AMX": "maximum absolute value over time",
}

# (base, names, staggered position codes, group key)
_GROUPS: list[tuple[int, int, tuple[str, ...], tuple[int, ...], str]] = [
    (GRNAM0, GCNAM0,
     ("XVEL", "YVEL", "S11", "S22", "S12", "ZVEL", "S33", "S23", "S31", "S31"),
     (1, 2, 0, 0, 3, 4, 0, 6, 5, 5), "grid"),
    (GCNAM0, STNAM0,
     ("DIL", "V-abs", "TauMx", "Sig-3", "Sig-1", "Sig-2", "M-Ang1", "ESS", "DEV",
      "aXVEL", "aYVEL", "aS12", "aZVEL", "aS23", "aS31", "aS31",
      "S.E.", "K.E.", "T.E.", "Fail", "e-plas"),
     (0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3), "computed"),
    (STNAM0, SCNAM0,
     ("t11_s1", "t11_s0", "t1v_s1", "t1v_s0", "t1sh", "t1d-r",
      "nvel-r", "ndis-r", "Sh-bnd", "T1slip", "TaSlip", "N-bnd",
      "t22_s1", "t22_s0", "t2v_s1", "t2v_s0", "t1t2s1",
      "t1t2s0", "t2sh", "t2d-r", "T2slip"),
     (0, 0, 8, 8, 8, 8, 0, 0, 8, 8, 0, 0, 0, 0, 9, 9, 10, 10, 9, 9, 9), "stope"),
    (SCNAM0, GMXNAM0,
     ("nd-s1", "nd-s0", "Tad-r", "t1d_s1", "t1d_s0", "t2d_s1", "t2d_s0",
      "stv_s1", "stv_s0", "sts_s1", "sts_s0"),
     (0,) * 11, "stope_computed"),
    (GMXNAM0, SMXNAM0, ("MaxVel",), (0,), "global_max"),
    (SMXNAM0, GACNAM0,
     ("nv-mx", "nd-mx", "t1v-mx", "t1d-mx", "t2v-mx", "t2d-mx", "t11-mx", "t22-mx"),
     (0,) * 8, "stope_max"),
    (GACNAM0, SACNAM0, ("Xdisp", "Ydisp", "Zdisp"), (0, 0, 0), "displacement"),
    (SACNAM0, MIXNAM0, ("",), (0,), "stope_acc"),
    (MIXNAM0, 255, ("skn", "sks", "sNs", "sSs"), (0, 0, 0, 0), "mixed"),
]

_STRESS = {"S11", "S22", "S33", "S12", "S23", "S31", "TauMx", "Sig-1", "Sig-2", "Sig-3", "ESS",
           "DEV", "aS12", "aS23", "aS31", "t11_s1", "t11_s0", "t22_s1", "t22_s0", "t1sh", "t2sh",
           "t1t2s1", "t1t2s0", "Sh-bnd", "N-bnd", "t11-mx", "t22-mx", "sNs", "sSs", "sts_s1", "sts_s0"}
_VELOCITY = {"XVEL", "YVEL", "ZVEL", "V-abs", "aXVEL", "aYVEL", "aZVEL", "MaxVel", "t1v_s1", "t1v_s0",
             "t2v_s1", "t2v_s0", "nvel-r", "nv-mx", "t1v-mx", "t2v-mx", "stv_s1", "stv_s0"}
_DISPLACEMENT = {"Xdisp", "Ydisp", "Zdisp", "t1d-r", "t2d-r", "ndis-r", "T1slip", "T2slip", "TaSlip",
                 "nd-s1", "nd-s0", "Tad-r", "t1d_s1", "t1d_s0", "t2d_s1", "t2d_s0", "nd-mx",
                 "t1d-mx", "t2d-mx"}
_ENERGY = {"S.E.", "K.E.", "T.E."}
_ANGLE = {"M-Ang1"}
_STIFFNESS = {"skn", "sks"}

LONG_NAMES: dict[str, str] = {
    "XVEL": "x velocity", "YVEL": "y velocity", "ZVEL": "z velocity",
    "S11": "normal stress σxx", "S22": "normal stress σyy", "S33": "normal stress σzz",
    "S12": "shear stress σxy", "S23": "shear stress σyz", "S31": "shear stress σzx",
    "DIL": "dilatation", "V-abs": "velocity magnitude", "TauMx": "maximum shear stress",
    "Sig-1": "principal stress σ1", "Sig-2": "principal stress σ2", "Sig-3": "principal stress σ3",
    "M-Ang1": "principal stress angle", "ESS": "excess shear stress", "DEV": "deviatoric stress",
    "S.E.": "strain energy", "K.E.": "kinetic energy", "T.E.": "total energy",
    "Fail": "failure indicator", "e-plas": "plastic strain", "MaxVel": "maximum velocity in model",
    "Xdisp": "x displacement", "Ydisp": "y displacement", "Zdisp": "z displacement",
    "T1slip": "slip along t1", "T2slip": "slip along t2", "TaSlip": "total slip",
    "ndis-r": "relative normal displacement", "nvel-r": "relative normal velocity",
    "Sh-bnd": "shear bond state", "N-bnd": "normal bond state",
}

# Staggered-grid position bitmask (VarPositions): 1 = half cell in i, 2 = in j, 4 = in k.
# Codes 8/9/10 are stope-relative (T1, T2, both) and resolved per plane in stagger_shift().
STAGGER_I, STAGGER_J, STAGGER_K = 1, 2, 4


@dataclass(frozen=True)
class VarInfo:
    code: int
    name: str
    group: str
    units: str
    position: int
    known: bool = True

    @property
    def long_name(self) -> str:
        return LONG_NAMES.get(self.name, self.name)

    @property
    def is_stope(self) -> bool:
        return self.group.startswith("stope") or self.group == "mixed"

    @property
    def is_velocity(self) -> bool:
        return self.name in _VELOCITY

    @property
    def is_stress(self) -> bool:
        return self.name in _STRESS


def _units_for(name: str) -> str:
    if name in _STRESS:
        return "Pa"
    if name in _VELOCITY:
        return "m/s"
    if name in _DISPLACEMENT:
        return "m"
    if name in _ENERGY:
        return "J"
    if name in _ANGLE:
        return "deg"
    if name in _STIFFNESS:
        return "Pa/m"
    return ""


def var_info(code: int) -> VarInfo:
    """Look up a WAVE variable number (without the function code)."""
    if code == FORM_NAME:
        return VarInfo(code, "Form", "formula", "", 0)
    for base, top, names, positions, group in _GROUPS:
        if base < code <= top:
            idx = code - base - 1
            if idx < len(names) and names[idx]:
                name = names[idx]
                return VarInfo(code, name, group, _units_for(name), positions[idx])
            break
    return VarInfo(code, f"VAR#{code}", "unknown", "", 0, known=False)


def code_for_name(name: str) -> int | None:
    """Reverse lookup (first match), case-insensitive."""
    lname = name.lower()
    for base, _top, names, _pos, _g in _GROUPS:
        for k, n in enumerate(names):
            if n and n.lower() == lname:
                return base + k + 1
    return None


def split_func(raw: int, base: int = 0xFFFF) -> tuple[int, int]:
    """Split a raw WAVE variable word into ``(var, func)``.

    The Pascal uses ``DIV $FFFF`` / ``MOD $FFFF`` (base 65535, DATA_RSC.PAS:700). No sample file
    uses a function code, so the base is configurable.
    """
    if raw < 0:
        return raw, 0
    return raw % base, raw // base


def func_name(func: int) -> str:
    return FUNC_NAMES[func] if 0 <= func < len(FUNC_NAMES) else f"F{func}"


def stagger_shift(position: int, ax1: str, ax2: str) -> tuple[int, int]:
    """Half-cell shifts ``(along ax1, along ax2)`` for a variable on a plane (``GetVarPos``)."""
    ax1, ax2 = ax1.lower(), ax2.lower()
    vpos = position
    if vpos == 8:
        vpos = 1 if ax1 == "i" else 2
    if vpos == 9:
        vpos = 4 if "k" in (ax1, ax2) else 2
    if vpos == 10:
        return 1, 1

    def bit(axis: str) -> int:
        if axis == "i":
            return vpos % 2
        if axis == "j":
            return (vpos % 4) // 2
        if axis == "k":
            return vpos // 4
        return 0

    return bit(ax1), bit(ax2)
