"""A zero-dependency web dashboard for watching a mission live.

Standard library only — no Flask, no websockets, nothing to `pip install`.
This is deliberately the "watch it work tonight, no installs" version; Phase
5 in the plan describes the fuller MQTT + Flask dashboard this is a stepping
stone toward, once there's a real Pi to run Mosquitto on.

Serves two pages against the same live data:
- `/` (scene.html) — the default landing page: a 3D view of the robot in
  the room, polling `/state.json`, `/room.json`, and `/pointcloud.json`.
  See scene.html's own docstring-equivalent comment for what "3D" does and
  doesn't mean here: `/room.json` is ground-truth geometry (a real robot
  has no such oracle), while `/pointcloud.json` is the opposite — an
  honestly-earned map built from repeated simulated depth scans, the same
  way the 2D occupancy grid is built from repeated ultrasonic readings, not
  read from the ground truth.
- `/index.html` — the 2D occupancy-grid map, polling `/state.json`. Not the
  default; reachable via scene.html's own nav link.

Both poll rather than push (~300ms) — simpler than a WebSocket, and at this
data rate and audience (one browser tab on the same machine or LAN) there is
nothing to gain from one.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

#: path -> (file on disk, content-type). Kept as a plain dict rather than a
#: general static-file server so nothing outside pi/web/ is ever reachable.
_STATIC_PAGES = {
    "/": ("scene.html", "text/html; charset=utf-8"),  # 3D scene is the default landing page
    "/index.html": ("index.html", "text/html; charset=utf-8"),  # 2D map, not the default
    "/scene.html": ("scene.html", "text/html; charset=utf-8"),
}

#: Applied to every response — defence in depth for a server that's safe by
#: construction anyway (see module docstring / docs/DEPLOY.md's security
#: review): the whole HTTP surface is GET-only, serves a fixed set of known
#: files by name (self.path never touches the filesystem), and the two JSON
#: endpoints call fixed, closed-over functions with no user input in the
#: request path they act on — there is nowhere for a request to inject a
#: path, a command, or a query. These headers narrow the browser-side blast
#: radius regardless: no embedding this page in someone else's iframe
#: (clickjacking), no MIME-sniffing a response into executing as something
#: it isn't, and a Content-Security-Policy that allows only this origin plus
#: the one CDN scene.html actually loads Three.js from — not a wildcard.
#:
#: script-src includes 'unsafe-inline': both pages' own logic IS an inline
#: <script> (confirmed by live testing — without this, the CSP blocks the
#: pages' own module scripts, not just third-party ones). This is a
#: deliberate, checked trade-off, not an oversight: every dynamic value
#: either page ever renders goes through .textContent (verified by grep —
#: nowhere does either file assign untrusted data via .innerHTML), so there
#: is no reflected-XSS path for an inline-script allowance to actually
#: enable. The alternative (a content hash per <script> block) would need
#: recomputing on every edit to either file and silently break, unnoticed,
#: exactly like this CSP did on its first real test, the moment anyone
#: forgot.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "connect-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'"
    ),
}


def make_handler(
    snapshot_fn: Callable[[], dict],
    room_fn: Optional[Callable[[], dict]] = None,
    pointcloud_fn: Optional[Callable[[], dict]] = None,
) -> type:
    """Build a request handler bound to the given data sources.

    A closure rather than constructor arguments because `http.server`
    instantiates the handler class itself for every request — there's no
    hook to pass extra arguments through.
    """

    def _send_security_headers(handler: BaseHTTPRequestHandler) -> None:
        for name, value in _SECURITY_HEADERS.items():
            handler.send_header(name, value)

    def _serve_json(handler: BaseHTTPRequestHandler, fn: Callable[[], dict]) -> None:
        try:
            body = json.dumps(fn()).encode("utf-8")
            status = 200
        except Exception as exc:  # noqa: BLE001 - surface it to the browser, don't crash the server
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            status = 500
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        _send_security_headers(handler)
        handler.end_headers()
        handler.wfile.write(body)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # the default per-request console spam isn't useful here

        def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            if self.path in _STATIC_PAGES:
                self._serve_static(self.path)
            elif self.path == "/state.json":
                _serve_json(self, snapshot_fn)
            elif self.path == "/room.json":
                if room_fn is None:
                    self.send_error(404, "no room geometry configured for this session")
                else:
                    _serve_json(self, room_fn)
            elif self.path == "/pointcloud.json":
                if pointcloud_fn is None:
                    self.send_error(404, "no depth-scan point cloud configured for this session")
                else:
                    _serve_json(self, pointcloud_fn)
            else:
                self.send_error(404)

        def _serve_static(self, path: str) -> None:
            filename, content_type = _STATIC_PAGES[path]
            with open(os.path.join(_HERE, filename), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            _send_security_headers(self)
            self.end_headers()
            self.wfile.write(body)

    return Handler


def start(
    snapshot_fn: Callable[[], dict],
    room_fn: Optional[Callable[[], dict]] = None,
    pointcloud_fn: Optional[Callable[[], dict]] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ThreadingHTTPServer:
    """Start the dashboard on a background thread and return the server.

    `room_fn`, if given, powers `/scene.html`'s 3D room geometry (e.g.
    `lambda: room.to_dict()` from sim_world.SimulatedRoom). Without it,
    scene.html still works but renders the robot with no walls around it.

    `pointcloud_fn`, if given, powers `/scene.html`'s 3D depth-scan point
    cloud (e.g. `lambda: mission.point_cloud.to_dict()`). Without it, that
    endpoint 404s and the page simply doesn't render one.

    Call `.shutdown()` on the returned server to stop it.
    """
    server = ThreadingHTTPServer(
        (host, port), make_handler(snapshot_fn, room_fn, pointcloud_fn)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
