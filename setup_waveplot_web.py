"""One-step installer for WavePlot Web on Windows, Linux or macOS.

Usage::

    setup_waveplot_web.bat [options]          Windows (double-click works too)
    ./setup_waveplot_web.sh [options]         Linux / macOS
    python setup_waveplot_web.py [options]    either

Finds a Python 3.12+ (3.13 preferred: the exact pins in ``waveplot_web/constraints-py313.txt``
are for 3.13), creates ``waveplot_python_venv`` next to this file, installs WavePlot Web and its
dependencies into it and starts the app. Running it again is safe: an existing healthy venv is
reused and only missing packages are installed. Afterwards use ``start_waveplot`` to start the app.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import start_waveplot as sw

WEB = sw.WEB
CONSTRAINTS = WEB / "constraints-py313.txt"
MIN_VERSION = (3, 12)
PREFERRED = ["3.13", "3.12", "3.14"]


class SetupError(Exception):
    pass


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    say(f"\n== {msg}")


def run(cmd: list[str]) -> None:
    say("> " + (subprocess.list2cmdline(cmd) if sw.IS_WIN else shlex.join(cmd)))
    subprocess.run(cmd, check=True)


def python_version(cmd: list[str]) -> tuple[int, int] | None:
    """(major, minor) of the interpreter started by *cmd*, or None if it does not run."""
    try:
        out = subprocess.run([*cmd, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        major, minor = out.stdout.strip().split(".")
        return int(major), int(minor)
    except ValueError:
        return None


def install_hint() -> str:
    if sw.IS_WIN:
        return ("Install Python 3.13 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')\n"
                "or run:  winget install Python.Python.3.13")
    if sw.os_label() == "macOS":
        return "Install Python 3.13 from https://www.python.org/downloads/ or run:  brew install python@3.13"
    return ("Install Python 3.13 with your package manager, e.g.\n"
            "  Ubuntu/Debian:  sudo apt install python3.13 python3.13-venv\n"
            "                  (older Ubuntu: sudo add-apt-repository ppa:deadsnakes/ppa first)\n"
            "  Fedora/RHEL:    sudo dnf install python3.13\n"
            "or pass an interpreter explicitly with --python /path/to/python3.13")


def find_base_python(forced: str | None) -> tuple[list[str], tuple[int, int]]:
    """Pick the interpreter used to create the venv."""
    if forced:
        candidates = [[forced]]
    elif sw.IS_WIN:
        candidates = [["py", f"-{v}"] for v in PREFERRED] if shutil.which("py") else []
        candidates.append([sys.executable])
    else:
        candidates = [[exe] for exe in (shutil.which(f"python{v}") for v in PREFERRED) if exe]
        candidates += [[exe] for exe in (shutil.which("python3"), sys.executable) if exe]
    for cmd in candidates:
        ver = python_version(cmd)
        if ver and ver >= MIN_VERSION:
            return cmd, ver
        if forced:
            found = f"Python {ver[0]}.{ver[1]}" if ver else "nothing that runs"
            raise SetupError(f"--python {forced} is {found}; WavePlot needs Python 3.12 or newer.")
    raise SetupError("No Python 3.12 or newer was found on this computer.\n" + install_hint())


def venv_hint(ver: tuple[int, int]) -> str:
    if sw.IS_WIN or sw.os_label() == "macOS":
        return ""
    return (f"\nOn Ubuntu/Debian the venv module is a separate package:\n"
            f"  sudo apt install python{ver[0]}.{ver[1]}-venv")


def ensure_venv(venv: Path, base: list[str], base_ver: tuple[int, int], recreate: bool) -> tuple[int, int]:
    """Create *venv* if needed; returns the venv's Python version."""
    py = sw.venv_python(venv)
    if venv.exists() and sw.foreign_venv(venv) and not recreate:
        other = venv.parent / f"waveplot_venv_{sw.os_label().lower()}"
        raise SetupError(f"{venv} holds a virtual environment made on another operating system.\n"
                         f"Use a separate folder for {sw.os_label()}, e.g.  --venv {other}\n"
                         f"or pass --recreate to replace it.")
    if venv.exists() and recreate:
        say(f"Removing {venv}")
        shutil.rmtree(venv)
    if py.exists():
        ver = python_version([str(py)])
        if ver and ver >= MIN_VERSION:
            say(f"Reusing {venv} (Python {ver[0]}.{ver[1]})")
            return ver
        state = f"Python {ver[0]}.{ver[1]}, too old" if ver else "it no longer runs"
        raise SetupError(f"The existing environment {venv} is unusable ({state}).\n"
                         f"Run again with --recreate to rebuild it.")
    if venv.exists() and any(venv.iterdir()):
        raise SetupError(f"{venv} exists but is not a complete virtual environment.\n"
                         f"Run again with --recreate to rebuild it.")
    try:
        run([*base, "-m", "venv", str(venv)])
    except subprocess.CalledProcessError as e:
        raise SetupError("Could not create the virtual environment." + venv_hint(base_ver)) from e
    if not py.exists():
        raise SetupError(f"The virtual environment was created but {py} is missing." + venv_hint(base_ver))
    return base_ver


def check_not_running(venv: Path) -> None:
    """On Windows a running ``waveplot.exe`` is locked and pip fails half-way; detect that up front."""
    exe = venv / "Scripts" / "waveplot.exe"
    if not sw.IS_WIN or not exe.exists():
        return
    probe = exe.with_name("waveplot.exe.setup-check")
    try:
        exe.rename(probe)
        probe.rename(exe)
    except OSError as e:
        raise SetupError("WavePlot is running from this environment, so Windows will not let it be updated.\n"
                         "Stop it (Ctrl+C in its window, or close the window) and run setup again.") from e


def install(venv: Path, ver: tuple[int, int], dev: bool) -> None:
    check_not_running(venv)
    py = str(sw.venv_python(venv))
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    extras = "server,dev" if dev else "server"
    cmd = [py, "-m", "pip", "install", "--only-binary=:all:", "-e", f"{WEB}[{extras}]"]
    if ver == (3, 13) and CONSTRAINTS.exists():
        try:
            run([*cmd, "-c", str(CONSTRAINTS)])
            return
        except subprocess.CalledProcessError:
            say("\nWARNING: installing the tested package versions failed; retrying with the newest compatible ones.")
    run(cmd)


def smoke_check(venv: Path) -> None:
    run([str(sw.venv_python(venv)), "-c",
         "import waveplot, dash, waitress; print('waveplot', waveplot.__version__, 'is installed')"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install WavePlot Web into a virtual environment and start it.",
        epilog="Other options are passed on to start_waveplot / 'waveplot serve' (e.g. --port 8060).")
    parser.add_argument("--python", help="Python interpreter to build the venv with (default: auto-detect).")
    parser.add_argument("--venv", type=Path, default=sw.DEFAULT_VENV,
                        help=f"Virtual environment folder (default: {sw.DEFAULT_VENV.name}).")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the virtual environment.")
    parser.add_argument("--dev", action="store_true", help="Also install developer tools (pytest, ruff).")
    parser.add_argument("--no-start", action="store_true", help="Only install; do not start the app.")
    opts, extra = parser.parse_known_args()
    venv = opts.venv.expanduser().resolve()

    say(f"WavePlot Web setup on {sw.os_label()} ({sys.platform}), running under Python {sys.version.split()[0]}")
    say(f"Repository:          {sw.ROOT}")
    say(f"Virtual environment: {venv}")
    try:
        step("1/4  Finding Python 3.12+")
        base, base_ver = find_base_python(opts.python)
        say(f"Using {' '.join(base)} (Python {base_ver[0]}.{base_ver[1]})")

        step("2/4  Creating the virtual environment")
        ver = ensure_venv(venv, base, base_ver, opts.recreate)

        step("3/4  Installing WavePlot Web and its dependencies (first time: a few minutes)")
        install(venv, ver, opts.dev)

        step("4/4  Checking the installation")
        smoke_check(venv)
    except SetupError as e:
        say(f"\nSETUP FAILED: {e}")
        return 1
    except subprocess.CalledProcessError as e:
        say(f"\nSETUP FAILED: command exited with code {e.returncode} (see the messages above).")
        return 1

    sw.INPUT.mkdir(parents=True, exist_ok=True)
    start = "start_waveplot.bat" if sw.IS_WIN else "./start_waveplot.sh"
    venv_opt = "" if venv == sw.DEFAULT_VENV.resolve() else f" --venv {venv}"
    say("\nWavePlot Web is installed.")
    say(f"  Put case folders (*.map with their .snp/.hst/.dmp/... files) in {sw.INPUT}")
    say(f"  Start the app later with:  {start}{venv_opt}")
    if opts.no_start:
        return 0
    say("")
    return sw.main(venv, extra)


if __name__ == "__main__":
    sys.exit(main())
