# Architecture

```
io  →  model  →  science / formula  →  render  →  web | cli | export
```

Everything below `web` is plain Python (NumPy/SciPy/xarray/Plotly) and usable from notebooks.

## io — bytes to records and arrays

* `mapfile.read_map` parses the 128-byte `.map` records into typed dataclasses
  (`records.SnapRecord`, `HistRecord`, `DumpRecord`, `GeomRecord`, `CrackRecord`), validates the
  Fortran markers and accumulates per-file offsets. Problems become `FormatIssue`s, never
  exceptions; there are no capacity limits.
* `binsource.BinSource` reads float32 slices (memmap on Linux/macOS, `np.fromfile` on Windows so
  WAVE can still rewrite the files). `read_snapshot`, `read_history`, `read_dump` strip markers.
* `geo.read_geometry`, `crk.read_crack_dump`, `script.parse_script/parse_log/parse_outline`,
  `legacy_settings.read_legacy_settings`.
* `xr_backend` exposes a dataset as a lazy `xarray.DataTree` (`engine="wave"`).

See `formats.md` for the byte layouts.

## model — datasets

* `WaveDataset` (from `open_dataset`) groups snapshots into `SnapshotSeries` (same variable,
  plane and extent; ordered by time), histories into `Receiver`s, dumps into `DumpSeries`, and links
  stope-variable snapshots to their stope.
* `coords` defines the physical coordinates: zone *i* spans ((i−1)·d, i·d), values at zone centres
  plus half-cell staggering; `ax1` horizontal, `ax2` vertical (`j` up).
* `registry.DatasetRegistry` maps ids (`root/relative/stem`) to datasets with path checks, keeps a
  few open (keyed by path, mtime, size — so rewritten runs reload) and owns a byte-bounded LRU
  cache for derived arrays and evaluated traces.

## science — numerics

Pure functions on arrays. Where v407 differed, functions take `mode="standard" | "v407"`;
`compat.QUIRKS` names every reproduced quirk and results carry a `Provenance` listing the quirks
used (shown as a badge in the UI). Golden tests check v407 mode against loop transliterations.

## formula — history processing

* `ops` — the operation registry: name, title, category, arity, typed parameters, legacy syntax,
  documentation and implementation. It drives evaluation, the pipeline editor's forms and the
  generated reference in Help.
* `legacy.parse_legacy` — v407 `Parse` semantics (accumulator, right-recursive binary operators,
  parameter rules) with diagnostics.
* `expr.parse_expr` — modern syntax through `ast` with a strict whitelist (no `eval`).
* `printer.to_expr` — trees back to modern text (pipe syntax), so legacy formulas are shown and
  stored as modern expressions.
* `evaluate` — evaluates trees against a dataset (`Trace` objects carry their own axis).

## render — figures

Plotly builders: `snapshot_fig` (maps, banded colours, contours, wiggle lines, multi-panel),
`overlays` (geometry with halos, receivers), `dump_fig` (glyphs, Mohr circles), `history_fig`,
`fig3d`. Arrays are float32 and serialised by Plotly as base64 typed arrays.

## web — Dash app

* `app.create_app` registers pages (`pages/*.py`, each with `layout()` and
  `register_callbacks(app)`) and the `/api/v1` routes.
* `shell` — Mantine AppShell, header (dataset picker, numerics switch, theme), global stores
  (`store-ds`, `store-prefs`, `store-kbd`, `store-nav`) and `assets/keyboard.js`.
* Callbacks are thin: read small JSON from stores, fetch the dataset from the registry, call a
  render builder, return a figure. Arrays never travel through stores.
* Cross-page navigation (e.g. clicking a receiver opens its histories) goes through `store-nav`
  with a timestamp so each request is consumed once.

## export

`tables` (CSV, v407 ASC/TBL), `netcdf` (DataTree → h5netcdf), `images` (matplotlib figures and
MP4/GIF animations via imageio-ffmpeg, no browser needed).
