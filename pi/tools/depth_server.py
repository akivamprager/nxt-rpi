"""A small HTTP server exposing POST /estimate_depth — the "bigger machine"
side of RemoteDepthEstimator (see scout/depth_estimator.py). Run this on a
laptop or a beefier host; the Pi 3B only ever captures frames and calls it.

Standard library only for the HTTP layer, mirroring web/server.py's
zero-dependency approach: routing, request/response framing, and error
handling here have no ML dependency and are fully testable with a fake
`estimate_fn` (see pi/tests/test_depth_server.py). The actual model lives in
midas_backend.py, imported lazily only by main() — never by importing this
module or running its tests.

    python3 pi/tools/depth_server.py

Environment variables:
    DEPTH_SERVER_HOST   interface to bind (default 0.0.0.0 — this is meant
                         to be reachable from the Pi on the same network,
                         unlike web/server.py's local-first default).
    DEPTH_SERVER_PORT   port to listen on (default 8090).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout.depth_estimator import encode_response_depth  # noqa: E402

EstimateFn = Callable[[np.ndarray], np.ndarray]


def make_handler(estimate_fn: EstimateFn) -> type:
    """Build a request handler bound to the given model function — same
    closure-over-a-callable pattern as web/server.py's make_handler, and for
    the same reason: http.server instantiates the handler class itself per
    request, so there's no constructor to pass extra state through."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/estimate_depth":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self.send_error(400, "empty request body")
                return
            jpeg_bytes = self.rfile.read(length)

            import cv2  # lazy: only needed to decode the request body

            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self.send_error(400, "could not decode request body as JPEG")
                return

            try:
                depth = estimate_fn(frame)
            except Exception as exc:  # noqa: BLE001 - surface it, don't crash the server
                self.send_error(500, str(exc))
                return

            body = encode_response_depth(depth)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def start(
    estimate_fn: EstimateFn, host: str = "127.0.0.1", port: int = 8090
) -> ThreadingHTTPServer:
    """Start the depth server on a background thread and return it.
    Call `.shutdown()` on the returned server to stop it."""
    server = ThreadingHTTPServer((host, port), make_handler(estimate_fn))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> int:
    host = os.environ.get("DEPTH_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("DEPTH_SERVER_PORT", "8090"))

    from midas_backend import load_estimate_fn  # the real model — see its own docstring

    estimate_fn = load_estimate_fn()
    server = start(estimate_fn, host=host, port=port)
    print(f"Depth server listening on {host}:{port} — POST a JPEG to /estimate_depth.")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
