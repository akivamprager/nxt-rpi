"""Tests for depth_server.py's HTTP routing/framing — a fake `estimate_fn`
stands in for the real ML model, so none of this needs onnxruntime, torch,
or a model file installed. Same socket.socketpair() technique as
test_server.py: a real handler instance, no real network binding.
"""

from __future__ import annotations

import http.client
import io
import os
import socket
import struct
import sys
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from depth_server import make_handler  # noqa: E402


class _Response:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self.body = body

    def header(self, name):
        return self.headers.get(name)


def _do_post(handler_class, path: str, body: bytes) -> _Response:
    """Send a POST with a binary body through a real handler instance, no
    real socket binding — see the module docstring."""
    client_sock, server_sock = socket.socketpair()
    request = (
        f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii") + body

    def run_handler():
        handler_class(server_sock, ("127.0.0.1", 0), None)
        try:
            server_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    thread = threading.Thread(target=run_handler, daemon=True)
    thread.start()
    client_sock.sendall(request)

    chunks = []
    client_sock.settimeout(5.0)
    try:
        while True:
            chunk = client_sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except socket.timeout:
        pass
    thread.join(timeout=5.0)
    client_sock.close()

    raw = b"".join(chunks)
    response = http.client.HTTPResponse(_FakeSocketForParsing(raw))
    response.begin()
    resp_body = response.read()
    return _Response(response.status, dict(response.getheaders()), resp_body)


class _FakeSocketForParsing:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def makefile(self, *args, **kwargs):
        return self._buf


def _jpeg_bytes() -> bytes:
    import cv2

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_estimate_depth_returns_the_estimate_fn_result():
    def fake_estimate(frame_bgr):
        assert frame_bgr.shape[2] == 3
        return np.full((frame_bgr.shape[0], frame_bgr.shape[1]), 777.0, dtype="<f4")

    handler = make_handler(fake_estimate)
    r = _do_post(handler, "/estimate_depth", _jpeg_bytes())

    assert r.status == 200
    assert r.header("Content-Type") == "application/octet-stream"
    height, width = struct.unpack(">II", r.body[:8])
    depth = np.frombuffer(r.body[8:], dtype="<f4").reshape(height, width)
    assert (depth == 777.0).all()


def test_unknown_path_is_404():
    handler = make_handler(lambda frame: np.zeros((1, 1)))
    r = _do_post(handler, "/not_a_real_path", _jpeg_bytes())
    assert r.status == 404


def test_empty_body_is_rejected():
    handler = make_handler(lambda frame: np.zeros((1, 1)))
    r = _do_post(handler, "/estimate_depth", b"")
    assert r.status == 400


def test_malformed_jpeg_is_rejected():
    handler = make_handler(lambda frame: np.zeros((1, 1)))
    r = _do_post(handler, "/estimate_depth", b"not a jpeg at all")
    assert r.status == 400


def test_estimate_fn_exception_returns_500_not_a_crash():
    def broken_estimate(frame_bgr):
        raise RuntimeError("model not loaded")

    handler = make_handler(broken_estimate)
    r = _do_post(handler, "/estimate_depth", _jpeg_bytes())
    assert r.status == 500


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
