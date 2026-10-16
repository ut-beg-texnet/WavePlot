from __future__ import annotations

import json

import pytest

from tests.web.conftest import dash_call

SYN = "synthrun/run1"


def _syn_id(app):
    from waveplot.web import state

    return next(e.id for e in state.registry().list_datasets() if e.id.endswith(SYN))


def _a_id(app):
    from waveplot.web import state

    ids = [e.id for e in state.registry().list_datasets() if e.id.endswith("mt_slip5")]
    if not ids:
        pytest.skip("sample dataset A not available")
    return ids[0]


def test_pages_and_api(client, app):
    for url in ("/", "/snapshots", "/histories", "/dumps", "/geometry", "/compare", "/settings", "/help",
                "/_dash-layout", "/_dash-dependencies", "/api/v1/health", "/api/v1/datasets"):
        assert client.get(url).status_code == 200, url
    ds = _syn_id(app)
    info = client.get(f"/api/v1/info/{ds}").get_json()
    assert info["counts"]["dumps"] == 3 and "folder" not in info
    fr = client.get(f"/api/v1/frame/{ds}?series=S22|ij|k=5&n=1").get_json()
    assert fr["dtype"] == "f4" and fr["shape"] == [12, 20]


def test_path_traversal_rejected(client, app):
    from waveplot.web import state

    root = next(iter(state.registry().roots))
    for bad in (f"{root}/../../etc/passwd", f"{root}/..%2F..%2Fsecret", "nope/x", f"{root}/"):
        assert client.get(f"/api/v1/info/{bad}").status_code == 404
    with pytest.raises((PermissionError, KeyError)):
        state.registry().resolve(f"{root}/../../../../Windows/win")


def test_callback_ids_exist_in_layouts(app):
    import dash

    from waveplot.web.app import PAGES

    ids: set[str] = set()

    def collect(node):
        if node is None:
            return
        if isinstance(node, list | tuple):
            for c in node:
                collect(c)
            return
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            ids.add(cid)
        for attr in ("children", "header", "navbar", "label", "leftSection", "rightSection", "description"):
            if hasattr(node, attr):
                collect(getattr(node, attr))

    collect(app.layout)
    for page in dash.page_registry.values():
        collect(page["layout"]())
    missing = set()
    for _key, cb in app.callback_map.items():
        for item in cb["inputs"] + cb.get("state", []):
            if isinstance(item["id"], str) and not item["id"].startswith("{") and item["id"] not in ids:
                missing.add(item["id"])
    assert not missing, f"callback ids missing from layouts: {sorted(missing)}"
    assert len(PAGES) == 8


@pytest.mark.parametrize("rep", ["map", "contour", "lines", "map+contour"])
@pytest.mark.parametrize("mode", ["standard", "v407"])
def test_snapshot_render(client, app, rep, mode):
    ds = _a_id(app)
    vals = {"store-ds.data": ds, "snp-series.value": "S11|ik|j=50", "snp-frame.value": 5, "snp-prows.value": 1,
            "snp-pcols.value": 1, "snp-pstep.value": 1, "snp-pmode.value": "frames", "snp-plinked.checked": True,
            "snp-rep.value": rep, "snp-cmap.value": "balance", "snp-range.value": "robust", "snp-nlev.value": 11,
            "snp-method.value": "lin", "snp-overlays.value": ["materials", "sources", "receivers"],
            "snp-symmetric.checked": True, "store-prefs.data": {"mode": mode, "scheme": "light"},
            "snp-linedir.value": "h", "snp-linescale.value": 1.0}
    out = dash_call(client, app, "snp-graph.figure", vals)
    fig = out["response"]["snp-graph"]["figure"]
    assert len(fig["data"]) >= 2
    assert "t = " in out["response"]["snp-time"]["children"]


def test_snapshot_multipanel(client, app):
    ds = _a_id(app)
    vals = {"store-ds.data": ds, "snp-series.value": "S22|ij|k=250", "snp-frame.value": 0, "snp-prows.value": 2,
            "snp-pcols.value": 2, "snp-pstep.value": 3, "snp-pmode.value": "frames", "snp-plinked.checked": True,
            "snp-rep.value": "map", "store-prefs.data": {}}
    fig = dash_call(client, app, "snp-graph.figure", vals)["response"]["snp-graph"]["figure"]
    assert sum(1 for t in fig["data"] if t["type"] == "heatmap") == 4


def test_history_render_and_formula(client, app):
    ds = _syn_id(app)
    traces = {"ds": ds, "traces": [
        {"id": "a", "expr": "h1", "self": 1, "scale": "lin", "visible": True},
        {"id": "b", "expr": "h1 | integrate | bandpass(1, 10)", "self": 1, "scale": "lin", "visible": True},
        {"id": "c", "expr": "vector_magnitude(h=1)", "self": 1, "scale": "log", "visible": True},
        {"id": "d", "expr": "h99", "self": None, "scale": "lin", "visible": True},
    ]}
    vals = {"hst-traces.data": traces, "store-ds.data": ds, "hst-layout.value": "stacked", "hst-vlink.value": "own",
            "hst-hlink.value": "common", "store-prefs.data": {"mode": "v407"}, "hst-spec-kind.value": "fourier",
            "hst-spec-smooth.value": 1}
    out = dash_call(client, app, "hst-graph.figure", vals)["response"]
    assert len(out["hst-graph"]["figure"]["data"]) == 3          # h99 fails and is reported in its card
    assert "does not exist" in json.dumps(out["hst-list"])
    assert len(out["hst-spec-graph"]["figure"]["data"]) == 3


def test_history_add_legacy_formula(client, app):
    ds = _syn_id(app)
    vals = {"store-ds.data": ds, "hst-traces.data": {"ds": ds, "traces": []}, "hst-formula.value": "[1]I&f,1,10",
            "hst-add-formula.n_clicks": 1}
    out = dash_call(client, app, "hst-traces.data", vals, changed=["hst-add-formula.n_clicks"])["response"]
    t = out["hst-traces"]["data"]["traces"][0]
    assert t["expr"].startswith("h1 | integrate | bandpass") and t["legacy"] == "[1]I&f,1,10"


@pytest.mark.parametrize("kind,field", [("map", "ess"), ("map", "vm"), ("map", "ess_opt"), ("axes", "ess"),
                                        ("vel", "ess")])
def test_dump_render(client, app, kind, field):
    ds = _syn_id(app)
    vals = {"store-ds.data": ds, "dmp-plane.value": "ij|k=5", "dmp-frame.value": 1, "dmp-kind.value": kind,
            "dmp-field.value": field, "dmp-theta.value": 30, "dmp-phi.value": 30, "dmp-coh.value": 0,
            "dmp-dens.value": 11, "dmp-gscale.value": 1, "dmp-cmap.value": "balance", "dmp-range.value": "robust",
            "dmp-overlays.value": ["materials", "stopes"], "store-prefs.data": {"mode": "standard"}}
    out = dash_call(client, app, "dmp-graph.figure", vals)["response"]
    assert len(out["dmp-graph"]["figure"]["data"]) >= 2


def test_dump_mohr(client, app):
    ds = _syn_id(app)
    vals = {"dmp-graph.clickData": {"points": [{"x": 50.0, "y": 30.0}]}, "dmp-frame.value": 0, "dmp-phi.value": 30,
            "dmp-coh.value": 0, "store-ds.data": ds, "dmp-plane.value": "ij|k=5", "store-prefs.data": {}}
    out = dash_call(client, app, "dmp-probe-info.children", vals)["response"]
    assert "principal" in json.dumps(out["dmp-probe-info"])
    assert out["dmp-mohr"]["figure"]["data"]


def test_geometry_and_datasets(client, app):
    ds = _syn_id(app)
    out = dash_call(client, app, "geo-3d.figure", {"store-ds.data": ds, "geo-layers.value": ["materials", "stopes",
                                                                                              "sources", "receivers"],
                                                    "geo-plane.value": "S22|ij|k=5", "geo-plane-frame.value": 1,
                                                    "geo-camera.value": "top", "store-prefs.data": {}})
    fig = out["response"]["geo-3d"]["figure"]
    types = {t["type"] for t in fig["data"]}
    assert {"mesh3d", "scatter3d", "surface"} <= types
    assert fig["layout"]["scene"]["camera"]["projection"]["type"] == "orthographic"
    out = dash_call(client, app, "geo-traj.figure", {"store-ds.data": ds, "geo-traj-rec.value": "1",
                                                     "geo-traj-kind.value": "disp", "store-prefs.data": {}})
    assert len(out["response"]["geo-traj"]["figure"]["data"]) == 4
    out = dash_call(client, app, "dsp-detail.children", {"store-ds.data": ds, "store-prefs.data": {}})
    assert "Materials" in json.dumps(out["response"]["dsp-detail"]["children"]).replace("MATERIALS", "Materials")
