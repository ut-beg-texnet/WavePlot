"""Colormap registry shared by Plotly (web) and matplotlib (exports).

Includes perceptually uniform sequential maps, zero-centred diverging maps (recommended for
signed stresses and velocities), a cyclic map for angles, and the original v407 rainbow palettes
recovered from ``DSPRIM.PAS`` ("WavePlot classic").
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

# v407 palettes (6-bit VGA DAC values converted to 8-bit), DSPRIM.PAS
WP_CLASSIC16 = ["#412092", "#4120c2", "#2020f3", "#2082f3", "#20a2b2", "#20b292", "#20d282", "#31ff31",
                "#92c220", "#a27120", "#a24110", "#c21010", "#f31010", "#f3f300"]
WP_CLASSIC256 = [
    "#101031", "#101051", "#101061", "#101071", "#080882", "#080892", "#0000a2", "#0000aa", "#0000ba",
    "#0000d2", "#0000e3", "#0000f3", "#0000ff", "#0010f3", "#0020e3", "#0031d2", "#0041c2", "#0049ba",
    "#0051b2", "#0059aa", "#0061a2", "#00699a", "#007192", "#00758e", "#00798a", "#007d86", "#008286",
    "#008282", "#008682", "#00867d", "#008a79", "#008e75", "#009271", "#009a69", "#00a261", "#00aa59",
    "#00b251", "#00ba49", "#00c241", "#00d231", "#00e320", "#00f310", "#00ff00", "#10f300", "#20e300",
    "#31d200", "#41c200", "#49ba00", "#51b200", "#59aa00", "#61a200", "#699a00", "#719200", "#758e00",
    "#798a00", "#7d8600", "#828600", "#828200", "#868200", "#867d00", "#8a7900", "#8e7500", "#927100",
    "#9a6900", "#a26100", "#aa5900", "#b25100", "#ba4900", "#c24100", "#d23100", "#e32000", "#f31000",
    "#ff0000"]


@dataclass(frozen=True)
class CmapInfo:
    name: str
    label: str
    kind: str          # "sequential" | "diverging" | "cyclic" | "classic"
    source: str        # "mpl" | "plotly" | "cet" | "list"
    key: str


CMAPS: dict[str, CmapInfo] = {c.name: c for c in [
    CmapInfo("balance", "Balance (diverging)", "diverging", "plotly", "balance"),
    CmapInfo("RdBu_r", "Red–Blue (diverging)", "diverging", "mpl", "RdBu_r"),
    CmapInfo("coolwarm", "Cool–Warm (diverging)", "diverging", "cet", "coolwarm"),
    CmapInfo("PuOr_r", "Purple–Orange (diverging)", "diverging", "mpl", "PuOr_r"),
    CmapInfo("curl", "Curl (diverging)", "diverging", "plotly", "curl"),
    CmapInfo("viridis", "Viridis", "sequential", "mpl", "viridis"),
    CmapInfo("cividis", "Cividis (colour-blind safe)", "sequential", "mpl", "cividis"),
    CmapInfo("magma", "Magma", "sequential", "mpl", "magma"),
    CmapInfo("inferno", "Inferno", "sequential", "mpl", "inferno"),
    CmapInfo("fire", "Fire", "sequential", "cet", "fire"),
    CmapInfo("turbo", "Turbo (rainbow)", "sequential", "mpl", "turbo"),
    CmapInfo("rainbow4", "CET rainbow (uniform)", "sequential", "cet", "rainbow4"),
    CmapInfo("gray", "Grey", "sequential", "mpl", "gray"),
    CmapInfo("twilight", "Twilight (cyclic, angles)", "cyclic", "mpl", "twilight"),
    CmapInfo("wp_classic", "WavePlot classic (v407 256-colour)", "classic", "list", "WP_CLASSIC256"),
    CmapInfo("wp_classic16", "WavePlot classic (v407 16-colour)", "classic", "list", "WP_CLASSIC16"),
]}
DEFAULT_DIVERGING = "balance"
DEFAULT_SEQUENTIAL = "viridis"


def _to_hex(rgb) -> str:
    r, g, b = (round(255 * float(c)) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _parse_css(c: str) -> str:
    c = c.strip()
    if c.startswith("#"):
        return c.lower()
    if c.startswith("rgb"):
        vals = [float(v) for v in c[c.index("(") + 1:c.index(")")].split(",")[:3]]
        return "#{:02x}{:02x}{:02x}".format(*(round(v) for v in vals))
    return c


@functools.lru_cache(maxsize=64)
def get_colors(name: str, n: int = 256, reverse: bool = False) -> tuple[str, ...]:
    """``n`` hex colours sampled evenly from the colormap ``name``."""
    info = CMAPS.get(name)
    if info is None:
        base = _mpl_like(name)
    elif info.source == "list":
        base = WP_CLASSIC256 if info.key == "WP_CLASSIC256" else WP_CLASSIC16
        if info.key == "WP_CLASSIC16":   # discrete palette: keep its colours exactly
            cols = list(base)
            return tuple(reversed(cols)) if reverse else tuple(cols)
    elif info.source == "plotly":
        import plotly.colors as pc

        base = [_parse_css(c) for c in getattr(pc.cmocean, info.key)]
    elif info.source == "cet":
        import colorcet

        base = list(colorcet.palette[info.key])
    else:
        base = _mpl_like(info.key)
    cols = _resample(base, n)
    if reverse:
        cols = cols[::-1]
    return tuple(cols)


def _mpl_like(name: str) -> list[str]:
    import matplotlib as mpl

    cmap = mpl.colormaps[name]
    return [_to_hex(cmap(x)) for x in np.linspace(0, 1, 256)]


def _resample(colors: list[str], n: int) -> list[str]:
    if len(colors) == n:
        return list(colors)
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in colors], dtype=float)
    xs = np.linspace(0, 1, len(colors))
    xn = np.linspace(0, 1, n)
    out = np.stack([np.interp(xn, xs, rgb[:, k]) for k in range(3)], axis=1)
    return ["#{:02x}{:02x}{:02x}".format(*(round(v) for v in row)) for row in out]


def plotly_colorscale(name: str, reverse: bool = False, n: int = 64) -> list[list]:
    cols = get_colors(name, n, reverse)
    if len(cols) == 1:
        return [[0.0, cols[0]], [1.0, cols[0]]]
    return [[k / (len(cols) - 1), c] for k, c in enumerate(cols)]


def mpl_cmap(name: str, reverse: bool = False):
    from matplotlib.colors import ListedColormap

    return ListedColormap(list(get_colors(name, 256, reverse)), name=name + ("_r" if reverse else ""))


def is_diverging(name: str) -> bool:
    info = CMAPS.get(name)
    return info is not None and info.kind == "diverging"


def banded_colorscale(levels: np.ndarray, colors: list[str], zmin: float, zmax: float,
                      rule: str = "floor") -> list[list]:
    """Stepped Plotly colorscale so that raw ``z`` renders in discrete bands.

    Band ``k`` covers ``[L_k, L_{k+1})`` and gets ``colors[k]`` (``rule='floor'``). With
    ``rule='toward_zero'`` (v407) intervals below zero take the colour of their upper level.
    Values outside ``[L_0, L_{n-1}]`` take the end colours.
    """
    lev = np.asarray(levels, dtype=float)
    span = zmax - zmin
    if span <= 0 or len(lev) == 0:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    def pos(v: float) -> float:
        return float(np.clip((v - zmin) / span, 0.0, 1.0))

    edges = [zmin, *[float(v) for v in lev[1:]], zmax]
    stops: list[list] = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo and k < len(edges) - 2:
            continue
        if rule == "toward_zero":
            # interval [L_k, L_{k+1}) below zero belongs to level k+1, at/above zero to level k
            col_idx = k + 1 if hi <= 0 and k + 1 < len(colors) else k
        else:
            col_idx = k
        col = colors[min(col_idx, len(colors) - 1)]
        stops.append([pos(lo), col])
        stops.append([pos(hi), col])
    stops[0][0], stops[-1][0] = 0.0, 1.0
    # Plotly requires non-decreasing positions
    for k in range(1, len(stops)):
        stops[k][0] = max(stops[k][0], stops[k - 1][0])
    return stops
