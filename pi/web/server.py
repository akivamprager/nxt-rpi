"""A zero-dependency web dashboard for watching a mission live.

Standard library only — no Flask, no websockets, nothing to `pip install`.
This is deliberately the "watch it work tonight, no installs" version; Phase
5 in the plan describes the fuller MQTT + Flask dashboard this is a stepping
stone toward, once there's a real Pi to run Mosquitto on.

The browser polls /state.json every ~300ms rather than using a WebSocket —
simpler, and at this data rate and audience (one browser tab on the same
machine or LAN) there is nothing to gain from a push connection.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_PATH = os.path.join(_HERE, "index.html")


def make_handler(snapshot_fn: Callable[[], dict]) -> type:
    """Build a request handler bound to `snapshot_fn`.

    A closure rather than a constructor argument because `http.server`
    instantiates the handler class itself for every request — there's no
    hook to pass extra arguments through.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # the default per-request console spam isn't useful here

        def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            if self.path in ("/", "/index.html"):
                self._serve_index()
            elif self.path == "/state.json":
                self._serve_state()
            else:
                self.send_error(404)

        def _serve_index(self) -> None:
            with open(_INDEX_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_state(self) -> None:
            try:
                body = json.dumps(snapshot_fn()).encode("utf-8")
                status = 200
            except Exception as exc:  # noqa: BLE001 - surface it to the browser, don't crash the server
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def start(
    snapshot_fn: Callable[[], dict], host: str = "127.0.0.1", port: int = 8080
) -> ThreadingHTTPServer:
    """Start the dashboard on a background thread and return the server.

    Call `.shutdown()` on the returned server to stop it.
    """
    server = ThreadingHTTPServer((host, port), make_handler(snapshot_fn))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
