"""Start the WavePlot web app from the virtual environment made by ``setup_waveplot_web``.

Usage (from anywhere)::

    start_waveplot.bat [options]            Windows
    ./start_waveplot.sh [options]           Linux / macOS
    python start_waveplot.py [options]      either

Serves ``waveplot_web/input_data`` with waitress. Any options are passed straight to
``waveplot serve`` (e.g. ``--port 8060``, ``--host 0.0.0.0``, ``--no-open``, more data folders).
``--venv PATH`` (or the ``WAVEPLOT_VENV`` environment variable) selects a non-default venv.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "waveplot_web"
INPUT = WEB / "input_data"
DEFAULT_VENV = ROOT / "waveplot_python_venv"
IS_WIN = os.name == "nt"


def os_label() -> str:
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "macOS"}.get(platform.system(), platform.system())


def venv_python(venv: Path) -> Path:
    """The interpreter inside *venv* for the current OS."""
    return venv / "Scripts" / "python.exe" if IS_WIN else venv / "bin" / "python"


def foreign_venv(venv: Path) -> bool:
    """True if *venv* was created by the other OS (e.g. a Windows venv seen from WSL)."""
    if IS_WIN:
        return (venv / "bin" / "python").exists() and not (venv / "Scripts").exists()
    return (venv / "Scripts" / "python.exe").exists() and not (venv / "bin").exists()


def setup_command() -> str:
    return "setup_waveplot_web.bat" if IS_WIN else "./setup_waveplot_web.sh"


def _pop_venv(args: list[str]) -> tuple[Path | None, list[str]]:
    rest: list[str] = []
    venv = None
    it = iter(args)
    for a in it:
        if a == "--venv":
            venv = Path(next(it, ""))
        elif a.startswith("--venv="):
            venv = Path(a.split("=", 1)[1])
        else:
            rest.append(a)
    return venv, rest


def main(venv: Path | None = None, args: list[str] | None = None) -> int:
    if args is None:
        venv_arg, args = _pop_venv(sys.argv[1:])
        venv = venv or venv_arg
    if venv is None:
        venv = Path(os.environ["WAVEPLOT_VENV"]) if os.environ.get("WAVEPLOT_VENV") else DEFAULT_VENV
    venv = venv.expanduser().resolve()
    py = venv_python(venv)

    if foreign_venv(venv):
        print(f"The virtual environment {venv} was created on another operating system.\n"
              f"Run {setup_command()} --venv <new folder> to make one for {os_label()}.", file=sys.stderr)
        return 1
    if not py.exists():
        print(f"WavePlot is not installed yet (no {py}).\nRun {setup_command()} first.", file=sys.stderr)
        return 1

    INPUT.mkdir(parents=True, exist_ok=True)
    cmd = [str(py), "-m", "waveplot", "serve", "--prod", str(INPUT), *args]
    print(f"Starting WavePlot on {os_label()}; case folders are read from {INPUT}")
    print("Press Ctrl+C to stop.", flush=True)
    if not IS_WIN:
        # Become the server process, so kill/systemd signals reach waitress directly.
        os.execv(str(py), cmd)
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        try:
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            proc.terminate()
        print("WavePlot stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
