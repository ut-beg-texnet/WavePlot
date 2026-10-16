from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.utils
import pytest

from tests.conftest import SAMPLE_A
from tests.fixtures.synth import SynthDataset


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    """A small synthetic run with a 3-D dump series, geometry with a stope, and velocity histories."""
    root = tmp_path_factory.mktemp("synthroot")
    sd = SynthDataset()
    sd.geometry(props=[(3.82e10, 1.8225e10, 2500.0)], mats=[(1, 1, 20, 1, 12, 1, 9)],
                sources=[(3, 10, 10, 6, 6, 5, 5)], stopes=[(1, 3, (8, 14, 6, 6, 3, 7), None)],
                extent=(1, 20, 1, 12, 1, 9), d=(5.0, 5.0, 5.0), grid=(1, 21, 1, 13, 1, 10))
    rng = np.random.default_rng(0)
    for t in (0.1, 0.2, 0.3):
        data = rng.normal(size=(19, 11, 9)).astype(np.float32)
        data[..., 3:] *= 1e5
        sd.dump(data, t, extent=(2, 20, 2, 12, 5, 5), iqty=19, jqty=11, kqty=1, dx=5, dy=5, dz=5)
        sd.snapshot(rng.normal(size=(20, 12)), var=4, time=t, axes="ij", extent=(1, 20, 1, 12, 5, 5), dx=5, dy=5)
    tt = np.linspace(0, 1, 201)
    for var, f in ((1, 3.0), (2, 4.0), (6, 5.0)):
        sd.history(np.sin(2 * np.pi * f * tt) * np.exp(-3 * tt), var=var, t1=0.0, t2=1.0, pos=(5, 6, 5))
    sd.write(root / "synthrun", "run1")
    return root


@pytest.fixture(scope="session")
def app(synth_root):
    from waveplot.config import AppConfig
    from waveplot.web.app import create_app

    roots = [synth_root]
    if SAMPLE_A.is_file():
        roots.append(SAMPLE_A.parent)
    cfg = AppConfig.from_cli(roots, [])
    return create_app(cfg)


@pytest.fixture()
def client(app):
    return app.server.test_client()


def dash_call(client, app, output_key: str, values: dict, changed: list[str] | None = None):
    """POST a callback request like the browser does. ``values`` maps 'id.prop' → value."""
    key = next(k for k in app.callback_map if k == output_key or output_key in k)
    cb = app.callback_map[key]

    def spec(items):
        out = []
        for it in items:
            out.append({"id": it["id"], "property": it["property"],
                        "value": values.get(f"{it['id']}.{it['property']}")})
        return out

    outputs = [{"id": o.split(".")[0], "property": o.split(".")[1]} for o in
               key.strip(".").split("...")] if key.startswith("..") else {"id": key.split(".")[0],
                                                                          "property": key.split(".")[1]}
    payload = {"output": key, "outputs": outputs, "inputs": spec(cb["inputs"]), "state": spec(cb.get("state", [])),
               "changedPropIds": changed or [f"{cb['inputs'][0]['id']}.{cb['inputs'][0]['property']}"]}
    r = client.post("/_dash-update-component", data=json.dumps(payload, cls=plotly.utils.PlotlyJSONEncoder),
                    content_type="application/json")
    assert r.status_code in (200, 204), r.data[:2000]
    return r.get_json() if r.status_code == 200 else None
