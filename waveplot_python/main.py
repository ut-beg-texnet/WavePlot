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
    from waveplot_python.logger import setup_logging  # type: ignore
else:
    from . import launch
    from .logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="WavePlot GUI")
    parser.add_argument(
        "map_file",
        nargs="?",
        help="Optional path to a .map file to load on startup",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to waveplot_python/logs/waveplot_debug.log",
    )
    args = parser.parse_args()
    
    # Set up logging if debug flag is enabled
    setup_logging(debug=args.debug)
    
    map_path = None
    if args.map_file:
        path = Path(args.map_file)
        map_path = str(path.expanduser())
    launch(map_path, debug=args.debug)


if __name__ == "__main__":
    main()
