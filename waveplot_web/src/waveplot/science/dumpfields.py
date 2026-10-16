"""Scalar fields derived from dump planes (velocities + stress tensor)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import stress
from .compat import Mode


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    units: str
    needs_3d: bool = False
    uses_angles: bool = False


FIELDS: dict[str, FieldSpec] = {f.key: f for f in [
    FieldSpec("ess", "Excess shear stress ESS (plane θ, friction φ)", "Pa", uses_angles=True),
    FieldSpec("ess_opt", "Maximum ESS over all orientations (friction φ)", "Pa", uses_angles=True),
    FieldSpec("iso", "Max in-plane shear (isochromatics)", "Pa"),
    FieldSpec("smax", "In-plane principal σ max (most tensile)", "Pa"),
    FieldSpec("smin", "In-plane principal σ min (most compressive)", "Pa"),
    FieldSpec("mean2", "In-plane mean stress", "Pa"),
    FieldSpec("s1", "3-D principal σ max (most tensile)", "Pa", True),
    FieldSpec("s2", "3-D principal σ intermediate", "Pa", True),
    FieldSpec("s3", "3-D principal σ min (most compressive)", "Pa", True),
    FieldSpec("tau3", "3-D max shear (σmax − σmin)/2", "Pa", True),
    FieldSpec("mean3", "Mean stress (σ11+σ22+σ33)/3", "Pa", True),
    FieldSpec("vm", "von Mises equivalent stress", "Pa", True),
    FieldSpec("vmag", "In-plane velocity magnitude", "m/s"),
    FieldSpec("vmag3", "3-D velocity magnitude", "m/s", True),
    FieldSpec("s11", "σ11", "Pa"), FieldSpec("s22", "σ22", "Pa"), FieldSpec("s33", "σ33", "Pa", True),
    FieldSpec("s12", "σ12", "Pa"), FieldSpec("s23", "σ23", "Pa", True), FieldSpec("s13", "σ13", "Pa", True),
    FieldSpec("v1", "velocity along i", "m/s"), FieldSpec("v2", "velocity along j", "m/s"),
    FieldSpec("v3", "velocity along k", "m/s", True),
]}


def compute_field(key: str, comps: dict[str, np.ndarray], inplane: dict[str, np.ndarray], theta: float = 0.0,
                  friction: float = 30.0, cohesion: float = 0.0, mode: Mode = "standard") -> np.ndarray:
    s11, s22, s12 = inplane["s11"], inplane["s22"], inplane["s12"]
    if key == "ess":
        return stress.ess(s11, s22, s12, theta, friction, cohesion, mode)
    if key == "ess_opt":
        return stress.ess_optimal(s11, s22, s12, friction, cohesion)[0]
    if key == "iso":
        return stress.max_shear_2d(s11, s22, s12)
    if key in ("smax", "smin"):
        p = stress.principal_2d(s11, s22, s12, mode)
        return p.sigma_max if key == "smax" else p.sigma_min
    if key == "mean2":
        return stress.mean_stress(s11, s22)
    if key == "vmag":
        return np.hypot(inplane["u"], inplane["v"])
    if FIELDS[key].needs_3d and "s33" not in comps:
        raise ValueError(f"{FIELDS[key].label} needs a 3-D dump (9 components)")
    if key in ("s1", "s2", "s3", "tau3"):
        p = stress.principal_3d(comps["s11"], comps["s22"], comps["s33"], comps["s12"], comps["s23"], comps["s13"])
        if key == "tau3":
            return 0.5 * (p.values[..., 0] - p.values[..., 2])
        return p.values[..., {"s1": 0, "s2": 1, "s3": 2}[key]]
    if key == "mean3":
        return stress.mean_stress(comps["s11"], comps["s22"], comps["s33"])
    if key == "vm":
        return stress.von_mises(comps["s11"], comps["s22"], comps["s33"], comps["s12"], comps["s23"], comps["s13"])
    if key == "vmag3":
        return np.sqrt(comps["v1"] ** 2 + comps["v2"] ** 2 + comps["v3"] ** 2)
    if key in comps:
        return np.asarray(comps[key], float)
    raise KeyError(key)
