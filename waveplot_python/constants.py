"""Shared constants mirroring the Turbo Pascal definitions.

This module provides constants and utility functions for mapping numeric variable IDs
from the Turbo Pascal WAVE format to human-readable names. The constants replicate
the variable group offsets defined in the original DATA_RSC.PAS file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Variable group offsets replicate DATA_RSC.PAS
# These offsets define the starting ID for each variable group category
GRNAM0 = 0      # Grid variables (velocities, stresses)
GCNAM0 = 20     # Grid computed variables (dilatation, energy, etc.)
STNAM0 = 50     # Surface traction variables
SCNAM0 = 100    # Surface computed variables
GMXNAM0 = 150   # Grid maximum variables
SMXNAM0 = 175   # Surface maximum variables
GACNAM0 = 200   # Grid accumulated variables (displacements)
SACNAM0 = 225   # Surface accumulated variables
MIXNAM0 = 235   # Mixed/interface variables
FORMNAME = 255  # Formula variable ID

# Function names for history/snapshot processing operations
# Index 0 is empty string (no function), indices 1-5 correspond to operations
VFUNC_NAMES: List[str] = ["", "MX", "mn", "Avg", "Abs", "AMX"]

# Lookup table mapping numeric variable IDs to human-readable names
# Keys are integers (variable IDs), values are strings (variable names)
# Variable IDs are computed as: group_offset + index + 1
# Example: GRNAM0=0, list=["XVEL", "YVEL"] -> {1: "XVEL", 2: "YVEL"}
_VAR_TABLE: Dict[int, str] = {
    **{GRNAM0 + i + 1: name for i, name in enumerate([
        "XVEL", "YVEL", "S11", "S22", "S12", "ZVEL", "S33", "S23", "S31", "S31",
    ])},
    **{GCNAM0 + i + 1: name for i, name in enumerate([
        "DIL", "V-abs", "TauMx", "Sig-3", "Sig-1", "Sig-2", "M-Ang1", "ESS", "DEV",
        "aXVEL", "aYVEL", "aS12", "aZVEL", "aS23", "aS31", "aS31", "S.E.", "K.E.",
        "T.E.", "Fail", "e-plas",
    ])},
    **{STNAM0 + i + 1: name for i, name in enumerate([
        "t11_s1", "t11_s0", "t1v_s1", "t1v_s0", "t1sh", "t1d-r",
        "nvel-r", "ndis-r", "Sh-bnd", "T1slip", "TaSlip", "N-bnd",
        "t22_s1", "t22_s0", "t2v_s1", "t2v_s0", "t1t2s1",
        "t1t2s0", "t2sh", "t2d-r", "T2slip",
    ])},
    **{SCNAM0 + i + 1: name for i, name in enumerate([
        "nd-s1", "nd-s0", "Tad-r", "t1d_s1", "t1d_s0", "t2d_s1", "t2d_s0",
        "stv_s1", "stv_s0", "sts_s1", "sts_s0",
    ])},
    **{GMXNAM0 + i + 1: name for i, name in enumerate(["MaxVel"])},
    **{SMXNAM0 + i + 1: name for i, name in enumerate([
        "nv-mx", "nd-mx", "t1v-mx", "t1d-mx", "t2v-mx", "t2d-mx", "t11-mx", "t22-mx",
    ])},
    **{GACNAM0 + i + 1: name for i, name in enumerate(["Xdisp", "Ydisp", "Zdisp"])},
    **{SACNAM0 + i + 1: name for i, name in enumerate([""])},
    **{MIXNAM0 + i + 1: name for i, name in enumerate(["skn", "sks", "sNs", "sSs"])},
    FORMNAME: "Form",
}


def var_name(vnum: int) -> str:
    """Return human-readable variable name for a Turbo Pascal variable ID.
    
    Args:
        vnum: Numeric variable ID from the WAVE format
        
    Returns:
        Human-readable variable name (e.g., "XVEL", "S11"), or "var{vnum}" 
        if not found, or empty string if vnum <= 0
    """
    return _VAR_TABLE.get(vnum, f"var{vnum}") if vnum > 0 else ""


def apply_var_split(value: int) -> tuple[int, int]:
    """Split combined Pascal integer into function ID and variable ID.
    
    In the Turbo Pascal format, a single integer can encode both a function
    ID (high 16 bits) and a variable ID (low 16 bits). This function extracts both.
    
    Args:
        value: Combined integer value containing func_id and var_id
        
    Returns:
        Tuple of (func_id, var_id) where:
        - func_id: High 16 bits (function operation index)
        - var_id: Low 16 bits (variable ID)
    """
    func_id = value // 0xFFFF
    var_id = value % 0xFFFF
    return func_id, var_id


def func_suffix(func_id: int) -> str:
    """Return suffix string for a history/snapshot function index.
    
    Function suffixes indicate how the data was processed (e.g., "MX" for maximum,
    "Avg" for average, "Abs" for absolute value).
    
    Args:
        func_id: Function operation index (0-5)
        
    Returns:
        Function suffix string (e.g., "MX", "Avg"), or empty string if invalid
    """
    return VFUNC_NAMES[func_id] if 0 <= func_id < len(VFUNC_NAMES) else ""
