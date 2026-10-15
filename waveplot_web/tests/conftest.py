from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SAMPLE_A = REPO / "Waveplot_exe" / "mt_slip5.map"


def _sample_b() -> Path | None:
    root = os.environ.get("WAVEPLOT_TEST_DATA_B")
    if not root:
        return None
    p = Path(root)
    maps = [p] if p.suffix.lower() == ".map" else sorted(p.glob("*.map"))
    return maps[0] if maps and maps[0].is_file() else None


@pytest.fixture(scope="session")
def sample_a_path() -> Path:
    if not SAMPLE_A.is_file():
        pytest.skip("sample dataset A (Waveplot_exe/mt_slip5) not available")
    return SAMPLE_A


@pytest.fixture(scope="session")
def ds_a(sample_a_path):
    from waveplot import open_dataset

    ds = open_dataset(sample_a_path)
    yield ds
    ds.close()


@pytest.fixture(scope="session")
def sample_b_path() -> Path:
    p = _sample_b()
    if p is None:
        pytest.skip("set WAVEPLOT_TEST_DATA_B to the cg11 dataset folder to run these tests")
    return p


@pytest.fixture(scope="session")
def ds_b(sample_b_path):
    from waveplot import open_dataset

    ds = open_dataset(sample_b_path)
    yield ds
    ds.close()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "ds_b" in getattr(item, "fixturenames", ()) or "sample_b_path" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.dataset_b)
