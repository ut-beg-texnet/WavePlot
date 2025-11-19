# WavePlot Python Architecture

This document sketches the high-level organisation of the WavePlot Python version.

## Goals

- Load the WAVE simulation outputs (`.map`, `.snp`, `.hst`, `.dmp`, `.geo`, etc.).
- Provide a desktop GUI for browsing snapshots, histories, dumps, geometry overlays, and derived plots.

## Code Layout

```
waveplot_python/
��� __init__.py           # Expose public entry points
��� main.py               # CLI entry script (supports `python main.py` or `python -m waveplot_python.main`)
��� constants.py          # Turbo Pascal constants and variable name lookups
��� map_parser.py         # Binary parser that turns .map tables into Python dataclasses
��� data_loader.py        # Readers that convert binary payloads into NumPy arrays
��� geometry_loader.py    # Reads .geo files (properties, materials, stopes, sources) into overlays
��� project.py            # High-level facade combining metadata, loaders, geometry, and formulas
��� gui/
    ��� __init__.py
    ��� app.py            # Tkinter + Matplotlib application shell
```

### `constants.py`
- Contains the static tables from `DATA_RSC.PAS` (variable IDs, function suffixes).
- Provides helpers for translating Pascal composite integers into human-friendly labels.

### `map_parser.py`
- Reads `.map` files record-by-record, mirroring the Pascal `SnapMapRec`, `HistMapRec`, etc.
- Produces dataclasses describing snapshots, histories, dumps, geometries, and crack data.
- Supplies metadata (offsets, dimensions, variable names) for all downstream consumers.

### `data_loader.py`
- Uses NumPy to read binary payloads (`.snp`, `.hst`, `.dmp`) using offsets from the parser.
- Returns NumPy arrays ready for plotting (2D grids, 1D time series, 3D volumes).
- Offers convenience helpers like `snapshot_series` and `time_axis`, and reports which companion files exist.

### `geometry_loader.py`
- Reads the `.geo` companions the Pascal version used (`DI_ReadGeom`).
- Extracts properties, materials, sources, and stopes while honouring the 128-byte block padding.
- Produces cached overlay data (`GeometryOverlay`) that highlights material blocks/boundaries, stopes, and source markers on 2D slices.

### `project.py`
- Binds metadata (`WaveMap`), loaders, and geometry into a single `WaveProject` facade.
- Exposes helper methods: `snapshot_array`, `history_series`, `dump_volume`, and geometry overlay queries (`snapshot_geometry_overlay`, `dump_geometry_overlay`).
- Implements the history formula system (integration, differentiation, FFTs, optional filtering).
- Keeps GUI logic clean by shielding it from binary layout details.

### `gui/app.py`
- Tkinter window with dataset tree, toolbar (colour map, dump component), and Matplotlib canvas.
- Reacts to tree selections, fetches data via `WaveProject`, and renders plots.
- Handles colour map selection, snapshot cycling, dump component choice, formula prompts, and colourbar min/max controls.
- Draws geometry overlays (material/stope outlines and source markers) on top of snapshots and dumps.

### `main.py`
- Small bootstrapper for launching the GUI.
- Works both when run as `python main.py` and as a module (`python -m waveplot_python.main`).

## Data Flow Overview

1. User selects a `.map` file.
2. `WaveProject` parses metadata (`map_parser.parse_map`) and prepares loaders (`WaveDataLoader`) and geometry accessors (`GeometryLoader`).
3. GUI populates the dataset tree with entries built from `WaveProject` records.
4. When a snapshot/history/dump is selected:
   - GUI asks `WaveProject` for the corresponding NumPy data.
   - If applicable, GUI requests a geometry overlay for the current plane.
   - Matplotlib renders the plot, applies colour scale updates, and draws overlays.
5. Optional: combine histories via formula entry, executed through `WaveProject.evaluate_history_formula`.

## Extensibility Pointers

- New binary record types: extend `map_parser.py` with another dataclass and parsing branch.
- Extra plotting modes: add new handlers in `gui/app.py`, reusing loaders from `WaveProject`.
- Geometry variants (labels, contouring stopes, 3D projections): extend `geometry_loader.py` and update `_draw_geometry_overlay` in the GUI.
