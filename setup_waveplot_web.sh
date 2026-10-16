#!/bin/sh
# Linux / macOS launcher: finds a Python 3 and runs setup_waveplot_web.py with the given options.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for c in python3.13 python3.12 python3.14 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        exec "$c" "$DIR/setup_waveplot_web.py" "$@"
    fi
done
echo "Python 3 was not found. Install Python 3.13, e.g.  sudo apt install python3.13 python3.13-venv" >&2
exit 1
