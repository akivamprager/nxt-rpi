"""Tests for depth_estimator.py: pose composition, pinhole backprojection,
and the remote-inference wire protocol — all exercised without a camera, a
Pi, a model file, or any real network socket (see RemoteDepthEstimator's
socketpair-based test below, the same technique test_server.py uses).
"""

from __future__ import annotations

import http.client
import math
import os
import socket
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout import depth_estimator as de  # noqa: E402
from scout.localize import CameraGeometry  # noqa: E402
from scout.pose2d import Pose2D  # noqa: E402
from scout.protocol import Telemetry  # noqa: E402
from scout.vision import CameraIntrinsics  # noqa: E402


# --------------------------------------------------------------- camera_pose_3d

def test_camera_pose_3d_at_turret_zero_extends_straight_ahead():
    geometry = CameraGeometry(
        camera_mount=Pose2D(40.0, 0.0, 0.0), turret_mount=Pose2D(20.0, 0.0, 0.0)
    )
    pose = de.camera_pose_3d(
        chassis_world=Pose2D(0.0, 0.0, 0.0),
        turret_angle_deg=0.0,
        geometry=geometry,
        camera_height_mm=150.0,
        camera_pitch_deg=-20.0,
    )
    assert pose.x_mm == 60.0  # 20 (turret_mount) + 40 (camera_mount), chassis facing +x
    assert pose.y_mm == 0.0
    assert pose.z_mm == 150.0
    assert pose.yaw_deg == 0.0
    assert pose.pitch_deg == -20.0


def test_camera_pose_3d_rotates_with_turret_angle():
    """Hand-computed: turret rotated 90 degrees swings the camera_mount
    offset from +x to +y, per pose2d.compose's rotation."""
    geometry = CameraGeometry(
        camera_mount=Pose2D(40.0, 0.0, 0.0), turret_mount=Pose2D(20.0, 0.0, 0.0)
    )
    pose = de.camera_pose_3d(
        chassis_world=Pose2D(0.0, 0.0, 0.0),
        turret_angle_deg=90.0,
        geometry=geometry,
        camera_height_mm=150.0,
        camera_pitch_deg=-20.0,
    )
    assert math.isclose(pose.x_mm, 20.0, abs_tol=1e-9)
    assert math.isclose(pose.y_mm, 40.0, abs_tol=1e-9)
    assert pose.yaw_deg == 90.0


def test_camera_pose_3d_follows_chassis_world_pose():
    geometry = CameraGeometry(
        camera_mount=Pose2D(0.0, 0.0, 0.0), turret_mount=Pose2D(0.0, 0.0, 0.0)
    )
    pose = de.camera_pose_3d(
        chassis_world=Pose2D(1000.0, 500.0, 45.0),
        turret_angle_deg=0.0,
        geometry=geometry,
        camera_height_mm=150.0,
        camera_pitch_deg=0.0,
    )
    assert pose.x_mm == 1000.0
    assert pose.y_mm == 500.0
    assert pose.yaw_deg == 45.0


# ----------------------------------------------------------------- intrinsics

def _intrinsics(fx=500.0, fy=500.0, cx=100.0, cy=80.0) -> CameraIntrinsics:
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return CameraIntrinsics(camera_matrix=matrix, dist_coeffs=np.zeros(5))


def _level_pose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0) -> de.CameraPose3D:
    return de.CameraPose3D(x, y, z, yaw, pitch)


# ------------------------------------------------------------- backproject_depth

def test_center_pixel_matches_sim_depth_scan_convention():
    """The whole point of the derivation in backproject_depth's docstring:
    a depth reading at the exact optical center, with the camera level and
    facing world +x, must land at (camera + depth, 0, 0) along +x — exactly
    what sim_world.py's depth_scan would produce for a level, zero-yaw ray."""
    intrinsics = _intrinsics(cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 100] = 1000.0  # exactly (cy, cx)

    points = de.backproject_depth(depth, intrinsics, _level_pose(), stride=1)

    assert len(points) == 1
    x, y, z = points[0]
    assert math.isclose(x, 1000.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(z, 0.0, abs_tol=1e-6)


def test_center_pixel_respects_camera_yaw_and_position():
    intrinsics = _intrinsics(cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 100] = 500.0

    pose = _level_pose(x=1000.0, y=2000.0, yaw=90.0)
    points = de.backproject_depth(depth, intrinsics, pose, stride=1)

    assert len(points) == 1
    x, y, z = points[0]
    # Facing +y now: the 500mm center ray lands entirely on y, none on x.
    assert math.isclose(x, 1000.0, abs_tol=1e-6)
    assert math.isclose(y, 2500.0, abs_tol=1e-6)
    assert math.isclose(z, 0.0, abs_tol=1e-6)


def test_center_pixel_respects_camera_pitch():
    intrinsics = _intrinsics(cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 100] = 1000.0

    pose = _level_pose(z=500.0, pitch=90.0)  # straight up
    points = de.backproject_depth(depth, intrinsics, pose, stride=1)

    assert len(points) == 1
    x, y, z = points[0]
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(z, 1500.0, abs_tol=1e-6)


def test_off_axis_pixel_offsets_in_the_expected_direction():
    """A pixel to the right of center (u > cx), camera level and facing +x:
    per this module's right-handed convention (verified against sim_world's
    yaw sign in the docstring), that should land on the -y side."""
    intrinsics = _intrinsics(fx=500.0, fy=500.0, cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 150] = 1000.0  # 50px right of center

    points = de.backproject_depth(depth, intrinsics, _level_pose(), stride=1)

    assert len(points) == 1
    x, y, z = points[0]
    assert x > 0.0
    assert y < 0.0  # right-of-center -> -y, matching camera_pose_3d's own
                     # turret-rotation test (turret +90 swings +x -> +y, so
                     # the camera's own "right" side faces -y at yaw 0)
    assert math.isclose(z, 0.0, abs_tol=1e-6)


def test_invalid_and_out_of_range_depths_are_skipped():
    intrinsics = _intrinsics()
    depth = np.array([
        [np.nan, 0.0, -5.0],
        [10.0, 5000.0, 1500.0],
    ])
    points = de.backproject_depth(depth, intrinsics, _level_pose(), stride=1, max_range_mm=3000.0)
    # Only the 10.0 and 1500.0 entries are finite, positive, and in range.
    assert len(points) == 2


def test_stride_samples_a_subset_of_pixels():
    intrinsics = _intrinsics(cx=2.0, cy=2.0)
    depth = np.full((6, 6), 500.0)
    points_full = de.backproject_depth(depth, intrinsics, _level_pose(), stride=1)
    points_strided = de.backproject_depth(depth, intrinsics, _level_pose(), stride=2)
    assert len(points_full) == 36
    assert len(points_strided) == 9  # ceil(6/2) x ceil(6/2)


# ------------------------------------------------------- backproject_depth_colorized

def test_backproject_depth_colorized_samples_color_at_the_right_pixel():
    intrinsics = _intrinsics(cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 100] = 1000.0  # optical center
    depth[80, 150] = 800.0   # a second, off-axis pixel

    frame_bgr = np.zeros((161, 201, 3), dtype=np.uint8)
    frame_bgr[80, 100] = (10, 20, 200)   # BGR: blue=10 green=20 red=200
    frame_bgr[80, 150] = (255, 128, 0)   # BGR: blue=255 green=128 red=0

    points = de.backproject_depth_colorized(
        depth, frame_bgr, intrinsics, _level_pose(), stride=1
    )

    assert len(points) == 2
    by_x = {round(p[0]): p for p in points}
    # RGB order in the output (see backproject_depth_colorized's b, g, r
    # unpacking from the BGR frame).
    assert by_x[1000][3:] == (200, 20, 10)
    center_ray_x = round(next(p[0] for p in points if p[3:] == (0, 128, 255)))
    assert center_ray_x != 1000  # the off-axis point landed at a different x


def test_backproject_depth_colorized_matches_backproject_depth_geometry():
    """Same geometry, just with colour attached — cross-check against the
    plain backproject_depth's already-verified output for the same inputs."""
    intrinsics = _intrinsics(cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 100] = 1000.0
    frame_bgr = np.full((161, 201, 3), 50, dtype=np.uint8)

    plain = de.backproject_depth(depth, intrinsics, _level_pose(), stride=1)
    colorized = de.backproject_depth_colorized(depth, frame_bgr, intrinsics, _level_pose(), stride=1)

    assert len(plain) == len(colorized) == 1
    assert plain[0] == colorized[0][:3]


# ------------------------------------------------------------- wire protocol

def test_decode_response_depth_round_trips():
    depth = np.array([[100.0, 200.0], [300.0, 400.0]], dtype="<f4")
    header = struct.pack(">II", *depth.shape)
    body = header + depth.tobytes()

    decoded = de.decode_response_depth(body)

    assert decoded.shape == (2, 2)
    assert list(decoded.flatten()) == [100.0, 200.0, 300.0, 400.0]


def test_decode_response_depth_rejects_a_too_short_body():
    try:
        de.decode_response_depth(b"\x00\x00\x00")
    except ValueError:
        pass
    else:
        raise AssertionError("a body shorter than the header should be rejected")


def test_decode_response_depth_rejects_a_length_mismatch():
    header = struct.pack(">II", 2, 2)  # claims 2x2 = 4 floats = 16 bytes
    body = header + b"\x00" * 8  # only provides 2 floats
    try:
        de.decode_response_depth(body)
    except ValueError:
        pass
    else:
        raise AssertionError("a body/header length mismatch should be rejected")


def test_encode_request_jpeg_produces_valid_jpeg_bytes():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    encoded = de.encode_request_jpeg(frame)
    assert encoded[:2] == b"\xff\xd8"  # JPEG SOI marker


# --------------------------------------------------------- RemoteDepthEstimator

class _FakeDepthServerHandler(BaseHTTPRequestHandler):
    """A minimal stand-in for depth_server.py's real handler — reads
    whatever JPEG body was sent (unused, this fake doesn't run a model) and
    replies with a canned 2x2 depth map in the real wire format."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body
        depth = np.array([[500.0, 600.0], [700.0, 800.0]], dtype="<f4")
        body = struct.pack(">II", *depth.shape) + depth.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


def test_remote_depth_estimator_round_trips_through_a_fake_server():
    """No real network binding anywhere: socket.socketpair() connects the
    fake server's handler directly to an http.client.HTTPConnection whose
    `.sock` is set by hand, bypassing HTTPConnection.connect()'s real
    socket.create_connection() call entirely — the client-side mirror of
    test_server.py's server-side technique."""
    client_sock, server_sock = socket.socketpair()

    def run_server() -> None:
        _FakeDepthServerHandler(server_sock, ("127.0.0.1", 0), None)
        try:
            server_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    conn = http.client.HTTPConnection("localhost", 0)
    conn.sock = client_sock
    estimator = de.RemoteDepthEstimator(
        "localhost", 0, connection_factory=lambda: conn
    )

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    depth = estimator.estimate(frame)
    thread.join(timeout=5.0)

    assert depth.shape == (2, 2)
    assert depth[0, 0] == 500.0
    assert depth[1, 1] == 800.0


def test_remote_depth_estimator_raises_on_a_non_200_response():
    client_sock, server_sock = socket.socketpair()

    class _ErrorHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_error(500, "model not loaded")

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass

    def run_server() -> None:
        _ErrorHandler(server_sock, ("127.0.0.1", 0), None)
        try:
            server_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    conn = http.client.HTTPConnection("localhost", 0)
    conn.sock = client_sock
    estimator = de.RemoteDepthEstimator("localhost", 0, connection_factory=lambda: conn)

    try:
        estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8))
    except ValueError as exc:
        assert "500" in str(exc)
    else:
        raise AssertionError("a non-200 response should raise")
    finally:
        thread.join(timeout=5.0)


# ------------------------------------------------------------- make_depth_scanner

def _telemetry(x_mm=0.0, y_mm=0.0, heading_deg=0.0, turret_deg=0.0) -> Telemetry:
    return Telemetry(0, x_mm, y_mm, heading_deg, turret_deg, 0, 0, 0, 0, 8000, 0)


class _FakeEstimator:
    def __init__(self, depth_mm: np.ndarray):
        self.depth_mm = depth_mm
        self.frames_seen: list[np.ndarray] = []

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.frames_seen.append(frame_bgr)
        return self.depth_mm


def test_make_depth_scanner_composes_capture_estimate_and_backproject():
    """An integration test tying the whole real-hardware chain together:
    capture -> estimate -> camera_pose_3d -> backproject_depth, using fakes
    for capture and estimation but the real pose/backprojection math — the
    same math test_center_pixel_matches_sim_depth_scan_convention already
    verified in isolation."""
    intrinsics = _intrinsics(cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 100] = 1000.0  # exactly the optical center
    estimator = _FakeEstimator(depth)

    frame_calls = []

    def capture_frame():
        frame_calls.append(1)
        return np.zeros((161, 201, 3), dtype=np.uint8)

    geometry = CameraGeometry(
        camera_mount=Pose2D(0.0, 0.0, 0.0), turret_mount=Pose2D(0.0, 0.0, 0.0)
    )
    scanner = de.make_depth_scanner(
        capture_frame, estimator, intrinsics, geometry,
        camera_height_mm=0.0, camera_pitch_deg=0.0, stride=1,
    )

    points = scanner(_telemetry(x_mm=500.0, y_mm=0.0, heading_deg=0.0))

    assert len(frame_calls) == 1
    assert len(estimator.frames_seen) == 1
    assert len(points) == 1
    x, y, z, r, g, b = points[0]
    # Chassis at (500, 0), zero-offset mount, level camera facing +x: the
    # 1000mm center ray lands 1000mm further along +x from the chassis.
    assert math.isclose(x, 1500.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(z, 0.0, abs_tol=1e-6)
    assert (r, g, b) == (0, 0, 0)  # the fake frame is all-black


def test_make_depth_scanner_follows_turret_rotation():
    intrinsics = _intrinsics(cx=100.0, cy=80.0)
    depth = np.full((161, 201), np.nan)
    depth[80, 100] = 500.0
    estimator = _FakeEstimator(depth)

    geometry = CameraGeometry(
        camera_mount=Pose2D(0.0, 0.0, 0.0), turret_mount=Pose2D(0.0, 0.0, 0.0)
    )
    scanner = de.make_depth_scanner(
        lambda: np.zeros((161, 201, 3), dtype=np.uint8),
        estimator, intrinsics, geometry,
        camera_height_mm=0.0, camera_pitch_deg=0.0, stride=1,
    )

    points = scanner(_telemetry(turret_deg=90.0))

    assert len(points) == 1
    x, y, z, r, g, b = points[0]
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 500.0, abs_tol=1e-6)


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
