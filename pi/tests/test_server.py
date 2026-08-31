"""Tests for the dashboard's HTTP handler — routing, JSON endpoints, and the
security headers added for public deployment (see docs/DEPLOY.md).

http.server.BaseHTTPRequestHandler does its request handling synchronously
inside __init__ (it's a socketserver.StreamRequestHandler), so it needs a
real socket-like object to read the request from and write the response to.
socket.socketpair() gives a connected, in-process pair with no network stack
involved — no bind, no listen, nothing the sandbox's network policy would
ever see — which is the same technique CPython's own test_httpserver.py uses
to test handlers without a real server loop.
"""

import http.client
import io
import os
import socket
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "web"))

from server import make_handler  # noqa: E402


class _Response:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self.body = body

    def header(self, name):
        return self.headers.get(name)


def _do_request(handler_class, method: str, path: str) -> _Response:
    """Send an HTTP request through a real handler instance (no real socket
    binding — see the module docstring) and parse the response.

    BaseHTTPRequestHandler.__init__ handles the request synchronously, so
    the handler runs on a background thread while the main thread writes
    the request and reads the response concurrently through the socketpair
    — necessary because both ends have finite OS buffers, so a handler
    whose response is larger than that buffer would deadlock against a
    caller that writes first and only reads afterward.
    """
    client_sock, server_sock = socket.socketpair()
    request = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"

    def run_handler():
        handler_class(server_sock, ("127.0.0.1", 0), None)
        try:
            server_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    thread = threading.Thread(target=run_handler, daemon=True)
    thread.start()
    client_sock.sendall(request.encode("ascii"))

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
    body = response.read()
    return _Response(response.status, dict(response.getheaders()), body)


class _FakeSocketForParsing:
    """Wraps already-received response bytes so http.client can parse them
    without a real socket — http.client.HTTPResponse expects a socket-like
    object with makefile()."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def makefile(self, *args, **kwargs):
        return self._buf


HANDLER = make_handler(
    snapshot_fn=lambda: {"ok": True, "n": 1},
    room_fn=lambda: {"walls": []},
    pointcloud_fn=lambda: {"points": [[1.0, 2.0, 3.0]], "resolution_mm": 20.0},
)
HANDLER_NO_ROOM = make_handler(snapshot_fn=lambda: {"ok": True})


def test_index_serves_with_200_and_security_headers():
    r = _do_request(HANDLER, "GET", "/")
    assert r.status == 200
    assert r.header("Content-Type").startswith("text/html")
    assert r.header("X-Content-Type-Options") == "nosniff"
    assert r.header("X-Frame-Options") == "DENY"
    assert r.header("Referrer-Policy") == "no-referrer"
    assert "default-src 'self'" in r.header("Content-Security-Policy")
    assert b"<html" in r.body.lower()


def test_root_path_serves_scene_html_not_index_html():
    """The 3D scene is the default landing page, not the 2D map — see
    server.py's _STATIC_PAGES. "/" and "/scene.html" must be byte-identical;
    "/index.html" (the 2D map) must be a genuinely different file."""
    root = _do_request(HANDLER, "GET", "/")
    scene = _do_request(HANDLER, "GET", "/scene.html")
    index = _do_request(HANDLER, "GET", "/index.html")
    assert root.body == scene.body
    assert root.body != index.body


def test_scene_html_is_also_served():
    r = _do_request(HANDLER, "GET", "/scene.html")
    assert r.status == 200
    assert b"<html" in r.body.lower()


def test_index_html_is_reachable_at_its_own_path():
    r = _do_request(HANDLER, "GET", "/index.html")
    assert r.status == 200
    assert b"<html" in r.body.lower()


def test_state_json_returns_the_snapshot_fn_result():
    r = _do_request(HANDLER, "GET", "/state.json")
    assert r.status == 200
    assert r.header("Content-Type") == "application/json"
    assert r.header("Cache-Control") == "no-store"
    assert r.header("X-Content-Type-Options") == "nosniff"
    import json
    assert json.loads(r.body) == {"ok": True, "n": 1}


def test_room_json_returns_room_fn_result_when_configured():
    r = _do_request(HANDLER, "GET", "/room.json")
    assert r.status == 200
    import json
    assert json.loads(r.body) == {"walls": []}


def test_pointcloud_json_returns_pointcloud_fn_result_when_configured():
    r = _do_request(HANDLER, "GET", "/pointcloud.json")
    assert r.status == 200
    import json
    assert json.loads(r.body) == {"points": [[1.0, 2.0, 3.0]], "resolution_mm": 20.0}


def test_pointcloud_json_404s_when_not_configured():
    r = _do_request(HANDLER_NO_ROOM, "GET", "/pointcloud.json")
    assert r.status == 404


def test_room_json_404s_when_no_room_fn_configured():
    """A session with no room_fn (e.g. talking to real hardware, which has
    no ground-truth room to show) must not crash — a clean 404 instead."""
    r = _do_request(HANDLER_NO_ROOM, "GET", "/room.json")
    assert r.status == 404


def test_unknown_path_is_404_not_a_file_read_attempt():
    """The core of why this server has no path-traversal surface: unknown
    paths are rejected by dictionary lookup before any filesystem access,
    never interpolated into a path — try a classic traversal payload."""
    for path in ("/../../etc/passwd", "/etc/passwd", "/nonexistent", "/index.html.bak"):
        r = _do_request(HANDLER, "GET", path)
        assert r.status == 404, f"{path} should 404, got {r.status}"


def test_post_to_a_page_path_is_rejected():
    """do_POST only ever recognizes /reconstruct_mesh (see below) — any
    other path, including real GET-only endpoints, is a clean 404, not a
    state change of any kind."""
    r = _do_request(HANDLER, "POST", "/state.json")
    assert r.status == 404


def test_reconstruct_mesh_404s_when_not_configured():
    """Without mesh_reconstruct_python configured, this endpoint doesn't
    exist (see server.py's docstring on POST /reconstruct_mesh) — the
    default HANDLER fixture has no mesh_reconstruct_python, so this should
    503, not silently attempt to run a script that isn't there."""
    r = _do_request(HANDLER, "POST", "/reconstruct_mesh")
    assert r.status == 503


def test_reconstruct_mesh_runs_the_configured_interpreter_and_streams_the_result():
    """A fake 'interpreter' standing in for a real Open3D-equipped Python —
    proves the actual subprocess wiring (argv shape passed to
    mesh_reconstruct_python, finding --out, streaming the resulting file
    back as the response body) without needing Open3D or network access
    for a real mesh_reconstruct.py run. `mesh_reconstruct_python` is
    invoked as `[mesh_reconstruct_python, <real mesh_reconstruct.py path>,
    "--host", ..., "--out", out_path]`, so this fake executable — not a
    .py file run BY python, but a directly-executable stand-in for python
    itself — ignores the real script path argument and just honours
    --out, exactly mirroring what argv it would actually receive."""
    import stat
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_interpreter = os.path.join(tmp_dir, "fake_interpreter")
        with open(fake_interpreter, "w") as f:
            f.write(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "out_path = sys.argv[sys.argv.index('--out') + 1]\n"
                "with open(out_path, 'wb') as fh:\n"
                "    fh.write(b'fake ply contents')\n"
            )
        os.chmod(fake_interpreter, os.stat(fake_interpreter).st_mode | stat.S_IEXEC)

        handler = make_handler(
            snapshot_fn=lambda: {"ok": True},
            mesh_reconstruct_python=fake_interpreter,
        )
        r = _do_request(handler, "POST", "/reconstruct_mesh")
        assert r.status == 200
        assert r.header("Content-Type") == "application/octet-stream"
        assert "room_mesh.ply" in r.header("Content-Disposition")
        assert r.body == b"fake ply contents"


def test_snapshot_fn_exception_returns_500_not_a_crash():
    handler = make_handler(snapshot_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    r = _do_request(handler, "GET", "/state.json")
    assert r.status == 500
    import json
    assert "boom" in json.loads(r.body)["error"]


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
