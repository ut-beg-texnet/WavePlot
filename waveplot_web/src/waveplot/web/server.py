"""Running the Dash app (development server or waitress)."""

from __future__ import annotations

import socket
import threading
import webbrowser

from ..config import AppConfig


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def run(cfg: AppConfig, host: str = "127.0.0.1", port: int = 8050, open_browser: bool = True,
        prod: bool = False, threads: int = 8, debug: bool = False) -> None:
    from .app import create_app

    if port == 0:
        port = free_port(host)
    app = create_app(cfg)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"WavePlot serving {', '.join(f'{k}={v}' for k, v in cfg.data_roots.items())}")
    print(f"  open {url}", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    if prod:
        from waitress import serve

        serve(app.server, host=host, port=port, threads=threads)
    else:
        app.run(host=host, port=port, debug=debug, dev_tools_hot_reload=debug)
