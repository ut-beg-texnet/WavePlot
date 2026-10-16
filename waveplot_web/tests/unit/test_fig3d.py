"""3-D model figure: camera presets, stope outlines and fault-snapshot draping."""

from __future__ import annotations

import numpy as np
import pytest

from waveplot.render.fig3d import CAMERAS, _box_edges, model_figure


def _segments(xs):
    return sum(1 for v in xs if v is None)


def test_box_edges_box_and_plane():
    box = {"i": (0.0, 1.0), "j": (0.0, 2.0), "k": (0.0, 3.0)}
    assert _segments(_box_edges(box)[0]) == 12
    plane = {"i": (5.0, 5.0), "j": (0.0, 2.0), "k": (0.0, 3.0)}
    xs = _box_edges(plane)[0]
    assert _segments(xs) == 4
    assert {v for v in xs if v is not None} == {5.0}


@pytest.mark.parametrize("cam", list(CAMERAS))
def test_camera_presets(ds_a, cam):
    lay = model_figure(ds_a, {"stopes"}, camera=cam).layout
    assert lay.uirevision == f"model3d-{cam}"
    assert lay.scene.camera.projection.type == ("perspective" if cam == "iso" else "orthographic")
    hidden = [ax for ax in ("xaxis", "yaxis", "zaxis") if lay.scene[ax].showticklabels is False]
    assert hidden == ({"front": ["yaxis"], "side": ["xaxis"], "top": ["zaxis"]}.get(cam, []))


def test_cg11_stope_filled_and_outlined(ds_b):
    fig = model_figure(ds_b, {"stopes"})
    names = [t.name for t in fig.data]
    assert [t.type for t in fig.data if t.name == "stope 2"] == ["mesh3d"]
    assert "stope 2 outline" in names
    outline = next(t for t in fig.data if t.name == "stope 2 outline")
    assert {v for v in outline.x if v is not None} == {697.5}       # i = 280 face: (280-1) * 2.5 m
    assert "kn 1e+14" in outline.text and "friction 30°" in outline.text


def test_cg11_fault_snapshot_draped_on_stope(ds_b):
    fig = model_figure(ds_b, {"stopes"}, plane_series=ds_b.series["TaSlip|kj|i=280"])
    assert not [t for t in fig.data if t.type == "mesh3d"]          # stope drawn outline-only under the data
    surf = next(t for t in fig.data if t.type == "surface")
    assert np.all(np.asarray(surf.x) == 697.5)                      # exactly on the stope, no z-fighting
    assert surf.text[0][0].startswith("i 280 · j ")
    assert "constant" in surf.colorbar.title.text and surf.cmax > surf.cmin    # slip=off: TaSlip is all zero


def test_cg11_zone_snapshot_near_fault_hides_fill(ds_b):
    fig = model_figure(ds_b, {"stopes"}, plane_series=ds_b.series["ESS|kj|i=280"])
    surf = next(t for t in fig.data if t.type == "surface")
    assert np.all(np.asarray(surf.x) == 698.75)                     # zone-centred quantity stays at its cell centre
    assert not [t for t in fig.data if t.type == "mesh3d"]


def test_cg11_unrelated_plane_keeps_fill(ds_b):
    fig = model_figure(ds_b, {"stopes"}, plane_series=ds_b.series["S11|ij|k=61"])
    assert [t.type for t in fig.data if t.name == "stope 2"] == ["mesh3d"]
    surf = next(t for t in fig.data if t.type == "surface")
    assert np.all(np.asarray(surf.y) == 151.25)                     # k = 61 cell centre
    assert surf.text[0][0].endswith("· k 61")
