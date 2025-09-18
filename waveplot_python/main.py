"""CLI entry point for the WavePlot GUI."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    # Allow running ``python main.py`` directly by injecting parent on sys.path.
    import os
    import sys

    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from waveplot_python import launch  # type: ignore
else:
    from . import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="WavePlot GUI")
    parser.add_argument(
        "map_file",
        nargs="?",
        help="Optional path to a .map file to load on startup",
    )
    args = parser.parse_args()
    map_path = None
    if args.map_file:
        path = Path(args.map_file)
        map_path = str(path.expanduser())
    launch(map_path)


if __name__ == "__main__":
    main()
