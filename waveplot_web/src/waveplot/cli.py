"""Command-line interface: ``waveplot serve | info | convert | export | formula | settings``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="WavePlot — visualise WAVE simulation outputs in the browser.")
settings_app = typer.Typer(help="Preferences utilities.", no_args_is_help=True)
app.add_typer(settings_app, name="settings")


def _version(value: bool) -> None:
    if value:
        from . import __version__

        typer.echo(f"waveplot {__version__}")
        raise typer.Exit()


@app.callback()
def _root(version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                                help="Show version and exit.")] = False) -> None:
    """WavePlot command-line tools."""


@app.command()
def info(
    map_path: Annotated[Path, typer.Argument(help="Path to a .map file (or base name / folder).")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    section: Annotated[str, typer.Option(help="all | snapshots | series | histories | dumps")] = "all",
) -> None:
    """Summarise a WAVE dataset."""
    from .model.dataset import open_dataset

    ds = open_dataset(map_path)
    data = ds.info()
    if as_json:
        if section != "all":
            data = json.loads(ds.summary()[section].to_json(orient="records"))
        typer.echo(json.dumps(data, indent=2, default=str))
        return
    typer.echo(f"Dataset {data['name']}  ({data['folder']})")
    c = data["counts"]
    typer.echo(f"  snapshots {c['snapshots']} in {c['series']} series | histories {c['histories']} "
               f"({c['receivers']} receivers) | dumps {c['dumps']} in {c['dump_series']} planes | "
               f"geometries {c['geometries']} | crack dumps {c['crack_dumps']}")
    if data.get("grid_cells"):
        sp = data["spacing_m"]
        typer.echo(f"  grid {data['grid_cells']} cells, spacing i/j/k = {sp['i']:g}/{sp['j']:g}/{sp['k']:g} m"
                   + (f", timestep {data['timestep_s']:g} s" if data.get("timestep_s") else "")
                   + (f", final cycle {data['final_cycle'][0]} (t = {data['final_cycle'][1]:g} s)"
                      if data.get("final_cycle") else ""))
    for m in data.get("materials", []):
        typer.echo(f"  material {m['number']}: vp {m['vp']:g} m/s, vs {m['vs']:g} m/s, rho {m['rho']:g} kg/m3")
    tables = ds.summary()
    import pandas as pd

    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.max_rows", 60):
        for name in (["series", "histories", "dumps"] if section == "all" else [section]):
            df = tables[name]
            if len(df):
                typer.echo(f"\n{name.upper()}\n{df.to_string(index=False)}")
    for issue in data["issues"]:
        typer.echo(f"! {issue}", err=True)


@app.command()
def serve(
    paths: Annotated[list[Path] | None, typer.Argument(help="Data folders or .map files to offer.")] = None,
    data_root: Annotated[list[str] | None, typer.Option("--data-root", help="NAME=PATH (repeatable).")] = None,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port (0 = pick a free one).")] = 8050,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open a browser tab.")] = True,
    prod: Annotated[bool, typer.Option("--prod", help="Serve with waitress (multi-threaded).")] = False,
    threads: Annotated[int, typer.Option(help="Worker threads for --prod.")] = 8,
    debug: Annotated[bool, typer.Option("--debug", help="Dash debug mode with hot reload.")] = False,
) -> None:
    """Start the WavePlot web app."""
    from .config import AppConfig
    from .web.server import run

    cfg = AppConfig.from_cli(paths or [], data_root or [])
    run(cfg, host=host, port=port, open_browser=open_browser, prod=prod, threads=threads, debug=debug)


@app.command()
def convert(
    map_path: Annotated[Path, typer.Argument(help="Path to a .map file.")],
    to: Annotated[str, typer.Option("--to", help="Output format (netcdf).")] = "netcdf",
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Output file.")] = None,
    include: Annotated[str, typer.Option(help="Comma list: snapshots,histories,dumps,geometry")] =
        "snapshots,histories,dumps,geometry",
) -> None:
    """Convert a dataset to a self-describing format (netCDF4/HDF5)."""
    from .export.netcdf import to_netcdf
    from .model.dataset import open_dataset

    if to.lower() not in ("netcdf", "nc"):
        raise typer.BadParameter("only --to netcdf is supported")
    ds = open_dataset(map_path)
    target = out or ds.path.with_suffix(".nc")
    to_netcdf(ds, target, include=[s.strip() for s in include.split(",") if s.strip()],
              progress=lambda msg: typer.echo(msg))
    typer.echo(f"wrote {target}")


@app.command()
def formula(
    map_path: Annotated[Path, typer.Argument(help="Path to a .map file.")],
    expression: Annotated[str, typer.Argument(help="Legacy v407 formula or modern expression.")],
    trace: Annotated[int | None, typer.Option(help="History number of the trace slot (legacy 'self').")] = None,
    mode: Annotated[str, typer.Option(help="standard | v407")] = "standard",
    explain: Annotated[bool, typer.Option("--explain", help="Print the parsed pipeline.")] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Write result as CSV.")] = None,
) -> None:
    """Evaluate a history formula and print summary statistics (or write CSV)."""
    from .formula.evaluate import evaluate_text
    from .model.dataset import open_dataset

    ds = open_dataset(map_path)
    res = evaluate_text(ds, expression, self_ref=trace, mode=mode)  # type: ignore[arg-type]
    if explain:
        typer.echo(f"syntax: {res.syntax}\nexpression: {res.modern}")
    for d in res.diagnostics:
        typer.echo(f"! {d}", err=True)
    tr = res.trace
    if tr is None:
        raise typer.Exit(1)
    typer.echo(f"{tr.label}: n={len(tr.y)} x0={tr.t0:g} dx={tr.dt:g} {tr.x_unit} "
               f"min={tr.y.min():.6g} max={tr.y.max():.6g}")
    if out:
        from .export.tables import write_csv

        write_csv(out, [tr])
        typer.echo(f"wrote {out}")


@app.command()
def export(
    map_path: Annotated[Path, typer.Argument(help="Path to a .map file.")],
    series: Annotated[str, typer.Option(help="Snapshot series slug, e.g. 'S11|ik|j=50'.")],
    out: Annotated[Path, typer.Option("-o", "--out", help="Output file (.mp4 .gif .png .pdf .svg).")],
    frames: Annotated[str, typer.Option(help="Frame range 'a:b' (0-based, end exclusive) or 'all'.")] = "all",
    cmap: Annotated[str, typer.Option(help="Colormap name.")] = "RdBu_r",
    fps: Annotated[float, typer.Option(help="Frames per second for animations.")] = 8.0,
    dpi: Annotated[int, typer.Option(help="Resolution for raster output.")] = 150,
) -> None:
    """Export a snapshot series as an animation (mp4/gif) or a single frame figure (png/pdf/svg)."""
    from .export.images import export_series
    from .model.dataset import open_dataset

    ds = open_dataset(map_path)
    s = ds.series[series]
    if frames == "all":
        idx = list(range(len(s)))
    else:
        a, _, b = frames.partition(":")
        idx = list(range(int(a or 0), int(b) if b else len(s)))
    export_series(ds, s, out, frames=idx, cmap=cmap, fps=fps, dpi=dpi,
                  progress=lambda k, n: typer.echo(f"frame {k}/{n}", nl=False) if False else None)
    typer.echo(f"wrote {out}")


@settings_app.command("import-legacy")
def import_legacy(set_file: Annotated[Path, typer.Argument(help="WVPL407.SET file.")]) -> None:
    """Import preferences from a v407 WVPL407.SET file and print them as JSON."""
    from .io.legacy_settings import read_legacy_settings
    from .web.prefs import preferences_from_legacy, save_preferences

    legacy = read_legacy_settings(set_file)
    prefs = preferences_from_legacy(legacy)
    path = save_preferences(prefs)
    typer.echo(json.dumps(prefs.model_dump(), indent=2))
    typer.echo(f"saved to {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
