"""Publication-quality figures and animations with matplotlib (no browser needed)."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ..render.snapshot_fig import PanelData, SnapshotStyle, make_levels, value_range
from ..science.colormaps import get_colors, mpl_cmap
from ..science.levels import level_colors

AXIS_NAMES = {"i": "x", "j": "y", "k": "z", "t": "t"}


def _axis_label(a: str) -> str:
    return "time (s)" if a == "t" else f"{AXIS_NAMES.get(a, a)} (m)"


def draw_panel(ax, panel: PanelData, style: SnapshotStyle, ds=None, colorbar: bool = True, cax=None):
    """Draw one plane on a matplotlib axes; returns the mappable."""
    import matplotlib.colors as mcolors

    lo, hi = value_range(panel, style)
    z = np.abs(panel.z) if style.abs_values else panel.z
    zt = z.T
    x, y = panel.x, panel.y
    cmap = mpl_cmap(style.cmap, style.reverse)
    norm = mcolors.Normalize(lo, hi)
    levels = None
    if style.banded or style.representation != "map" or style.mode == "v407":
        levels = make_levels(lo, hi, style)
    if levels is not None and (style.banded or style.mode == "v407"):
        lev = np.asarray(levels.values, dtype=float)
        cols = level_colors(levels, list(get_colors(style.cmap, 256, style.reverse)),
                            "#ffffff" if style.mode == "v407" else None)
        bounds = np.concatenate([[min(lo, lev[0]) - 1e-30], lev[1:], [max(hi, lev[-1]) + 1e-30]])
        if np.all(np.diff(bounds) > 0):
            cmap = mcolors.ListedColormap(cols)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
    mappable = None
    if style.representation in ("map", "map+contour"):
        mappable = ax.pcolormesh(x, y, zt, cmap=cmap, norm=norm, shading="nearest",
                                 rasterized=True)
    if style.representation in ("contour", "map+contour") and levels is not None:
        from ..science.contours import contour_lines, pick_contour_levels

        clev = pick_contour_levels(levels.values, 40)
        lcols = level_colors(levels, list(get_colors(style.cmap, 256, style.reverse)))
        lv = list(levels.values)
        for lev_v, xs, ys in contour_lines(zt, np.asarray(x, float), np.asarray(y, float), clev,
                                           "v407" if style.mode == "v407" else style.contour_engine):
            k = int(np.argmin([abs(v - lev_v) for v in lv]))
            col = lcols[k] if style.representation == "contour" else "k"
            ax.plot(xs, ys, color=col, lw=0.8)
        if mappable is None:
            import matplotlib.cm as cm

            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    if style.representation == "lines":
        amax = max(abs(lo), abs(hi)) or 1.0
        rows = np.unique(np.linspace(0, len(y) - 1, min(len(y), style.max_lines)).round().astype(int))
        sp = abs(y[rows[1]] - y[rows[0]]) if len(rows) > 1 else 1.0
        sgn = 1 if panel.vertical_up else -1
        for r in rows:
            ax.plot(x, y[r] + sgn * style.line_scale * sp / amax * zt[r], color="k", lw=0.6)
    if ds is not None and panel.plane is not None and not panel.spacetime:
        _draw_geometry(ax, ds, panel, style)
    ax.set_xlabel(_axis_label(panel.ax1))
    ax.set_ylabel(_axis_label(panel.ax2))
    if not panel.spacetime:
        ax.set_aspect("equal")
    ax.set_xlim(x[0] - (x[1] - x[0]) / 2 if len(x) > 1 else x[0] - 1, x[-1] + (x[-1] - x[-2]) / 2 if len(x) > 1
                else x[0] + 1)
    yl = (y[0] - (y[1] - y[0]) / 2, y[-1] + (y[-1] - y[-2]) / 2) if len(y) > 1 else (y[0] - 1, y[0] + 1)
    ax.set_ylim(yl if panel.vertical_up else yl[::-1])
    ax.set_title(panel.title, fontsize=9, loc="left")
    if colorbar and mappable is not None:
        cb = ax.figure.colorbar(mappable, ax=ax if cax is None else None, cax=cax, shrink=0.85, pad=0.02)
        cb.set_label(f"{panel.label} ({panel.units})" if panel.units else panel.label)
    return mappable


def _draw_geometry(ax, ds, panel: PanelData, style: SnapshotStyle) -> None:
    from matplotlib.patches import Rectangle

    geom = ds.geometry
    plane = panel.plane
    if geom is None or plane is None:
        return
    layers = set(style.overlays)
    if "materials" in layers:
        for m in geom.materials:
            if not plane.intersects(m.extent):
                continue
            x0, x1 = plane.span(m.extent, plane.ax1)
            y0, y1 = plane.span(m.extent, plane.ax2)
            if geom.grid is not None and m.extent.range(plane.ax1) == geom.grid.range(plane.ax1):
                if m.extent.range(plane.ax2) == geom.grid.range(plane.ax2):
                    continue
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=0.6, ec="0.3"))
    if "stopes" in layers:
        for st in geom.stopes:
            if not plane.intersects(st.extent):
                continue
            x0, x1 = plane.span(st.extent, plane.ax1)
            y0, y1 = plane.span(st.extent, plane.ax2)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.6, ec="#c92a2a"))
    if "sources" in layers:
        for s in geom.sources:
            if plane.intersects(s.extent, tol=1):
                ax.plot(plane.center((s.extent.i1, s.extent.j1, s.extent.k1), plane.ax1),
                        plane.center((s.extent.i1, s.extent.j1, s.extent.k1), plane.ax2), "*", ms=9,
                        color="#0b7285", mec="w", mew=0.5)
    if "receivers" in layers:
        for r in ds.receivers:
            off = 0
            if plane.fixed_axis:
                off = abs({"i": r.position[0], "j": r.position[1], "k": r.position[2]}[plane.fixed_axis]
                          - (plane.fixed_index or 0))
            ax.plot(plane.center(r.position, plane.ax1), plane.center(r.position, plane.ax2), "v", ms=5,
                    mfc="k" if off <= 1 else "none", mec="k", mew=0.8)


def render_panel_bytes(panel: PanelData, style: SnapshotStyle, ds=None, fmt: str = "png", dpi: int = 200,
                       width_mm: float = 180) -> bytes:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    aspect = 0.75
    if not panel.spacetime and len(panel.x) > 1 and len(panel.y) > 1:
        aspect = min(max(abs(panel.y[-1] - panel.y[0]) / max(abs(panel.x[-1] - panel.x[0]), 1e-9), 0.25), 2.0)
    w = width_mm / 25.4
    fig = plt.figure(figsize=(w, w * aspect * 0.9 + 0.8), dpi=dpi, layout="constrained")
    ax = fig.add_subplot(111)
    draw_panel(ax, panel, style, ds)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def export_series(ds, series, out: str | Path, frames: Sequence[int] | None = None, style: SnapshotStyle | None = None,
                  cmap: str | None = None, fps: float = 8.0, dpi: int = 150,
                  progress: Callable[[int, int], None] | None = None) -> Path:
    """Write an animation (``.mp4`` / ``.gif``) or a single-frame figure (``.png/.pdf/.svg``)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from ..render.panels import snapshot_panel

    out = Path(out)
    style = style or SnapshotStyle(scheme="light")
    if cmap:
        style = style.model_copy(update={"cmap": cmap})
    if style.range_mode == "frame":
        style = style.model_copy(update={"range_mode": "series"})
    frames = list(range(len(series)) if frames is None else frames)
    ext = out.suffix.lower()
    if ext in (".png", ".pdf", ".svg"):
        out.write_bytes(render_panel_bytes(snapshot_panel(ds, series, frames[0]), style, ds, ext[1:], dpi))
        return out

    first = snapshot_panel(ds, series, frames[0])
    aspect = 0.75
    if not first.spacetime:
        aspect = min(max(abs(first.y[-1] - first.y[0]) / max(abs(first.x[-1] - first.x[0]), 1e-9), 0.3), 1.6)
    fig = plt.figure(figsize=(8, 8 * aspect * 0.9 + 0.9), dpi=dpi)
    images = []
    for k, f in enumerate(frames):
        fig.clf()
        ax = fig.add_subplot(111)
        draw_panel(ax, snapshot_panel(ds, series, f), style, ds)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        h, w = buf.shape[:2]
        images.append(buf[: h - h % 2, : w - w % 2])     # even dimensions for H.264
        if progress:
            progress(k + 1, len(frames))
    plt.close(fig)
    if ext == ".gif":
        from PIL import Image

        pil = [Image.fromarray(a) for a in images]
        pil[0].save(out, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0, optimize=True)
        return out
    import imageio_ffmpeg

    h, w = images[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(str(out), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=2, quality=8)
    writer.send(None)
    for a in images:
        writer.send(np.ascontiguousarray(a[:h, :w]))
    writer.close()
    return out
