"""Turning a single Pi-camera frame into 3D points in the world frame —
Phase 2's real-hardware counterpart to sim_world.py's simulated depth_scan.

A single Pi Camera Module has no depth channel, and the turret only yaws (see
docs/BUILD.md's "three servos total" constraint) — there's no second sensor
or scan axis to fall back on. What's left is monocular depth estimation: an
ML model predicts a per-pixel depth map from one RGB frame, and this module
turns that depth map into world-frame (x, y, z) points using the robot's own
telemetry-derived pose, the same way sim_world.py's depth_scan turns ray
casts into points — feeding the exact same PointCloudMap/mission.py plumbing
either way.

Two things are deliberately split into pure, hardware-free functions so they
can be fully tested without a camera, a Pi, or a model file:

- `camera_pose_3d` — where is the camera, in the world, right now? Reuses
  localize.CameraGeometry's existing camera_mount/turret_mount chain (see
  localize.py's docstring) and adds the two fixed measurements a 2D-only
  Pose2D chain has no room for: mounting height and downward tilt (see
  config.py's CameraGeometryConfig.camera_height_mm/camera_pitch_deg).
- `backproject_depth` — given a depth map and where the camera was, which
  world points does that imply? Standard pinhole backprojection, rotated
  into world coordinates via the exact same yaw/pitch convention
  sim_world.py's depth_scan already established (see that function's
  docstring and this module's own derivation below) — so a real depth map
  and a simulated ray-cast scan produce points in the identical frame.

The actual ML inference is a separate, swappable concern (`DepthEstimator`
below) — this module has no opinion on which model produces the depth map,
only on what a depth map means once you have one.
"""

from __future__ import annotations

import http.client
import math
import struct
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .localize import CameraGeometry
from .pose2d import Pose2D, compose
from .protocol import Telemetry
from .vision import CameraIntrinsics

Point3 = tuple[float, float, float]
ColoredPoint3 = tuple[float, float, float, int, int, int]


@dataclass(frozen=True)
class CameraPose3D:
    """Where the camera is and which way it's looking, in world mm/degrees.

    Extends Pose2D with the two things a floor-plane-only pose can't
    express: height off the floor and downward/upward tilt. `yaw_deg`
    follows this project's usual bearing convention (0 = facing +x,
    increasing counter-clockwise); `pitch_deg` follows sim_world.py's
    depth_scan convention (positive = tilted up)."""

    x_mm: float
    y_mm: float
    z_mm: float
    yaw_deg: float
    pitch_deg: float


def camera_pose_3d(
    chassis_world: Pose2D,
    turret_angle_deg: float,
    geometry: CameraGeometry,
    camera_height_mm: float,
    camera_pitch_deg: float,
) -> CameraPose3D:
    """Where the camera is right now, given the chassis's current world pose.

    Mirrors localize.marker_in_camera_from_chassis's chassis -> turret ->
    camera chain exactly (same compose() calls, same turret_pose
    construction — see that function for why turret_mount.heading_deg is
    replaced by turret_angle_deg rather than added to it), just without the
    final step into a marker's frame, and with height/pitch tacked on since
    those have no 2D representation at all.
    """
    turret_pose = Pose2D(geometry.turret_mount.x_mm, geometry.turret_mount.y_mm, turret_angle_deg)
    world_turret = compose(chassis_world, turret_pose)
    world_camera = compose(world_turret, geometry.camera_mount)
    return CameraPose3D(
        world_camera.x_mm, world_camera.y_mm, camera_height_mm,
        world_camera.heading_deg, camera_pitch_deg,
    )


def backproject_depth(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose: CameraPose3D,
    stride: int = 4,
    max_range_mm: float = 3000.0,
) -> list[Point3]:
    """Turn a depth map into world-frame points.

    `depth_mm` is an (H, W) array, one metric depth value per pixel (mm,
    straight-line distance along the camera's forward/optical axis — NOT
    Euclidean distance from the camera). Non-finite, non-positive, or
    out-of-range depths are skipped, so an estimator can freely mark unknown
    pixels as NaN or 0 rather than needing to filter them itself.

    `stride` samples every `stride`-th pixel in each axis rather than
    backprojecting every one — a depth map is usually far denser than the
    point cloud needs (PointCloudMap dedups down to one point per
    resolution cell anyway; see pointcloud.py), and this keeps the transform
    itself cheap on a Pi.

    Assumes `depth_mm` is already in intrinsics' own (undistorted) pixel
    grid — run lens-distortion correction on the frame before estimating
    depth from it, not after, if the model's input needs it.

    Rotation derivation (verified against sim_world.py's depth_scan, not
    just asserted — see test_depth_estimator.py's
    test_center_pixel_matches_sim_depth_scan_convention): a camera-local ray
    (x_c, y_c, z_c) = (right, down, forward) maps to a world direction via
        wx = (y_c*sin(pitch) + z_c*cos(pitch)) * cos(yaw) + x_c*sin(yaw)
        wy = (y_c*sin(pitch) + z_c*cos(pitch)) * sin(yaw) - x_c*cos(yaw)
        wz = z_c*sin(pitch) - y_c*cos(pitch)
    which for the centre ray (x_c=y_c=0, z_c=1) reduces to
    (cos(pitch)cos(yaw), cos(pitch)sin(yaw), sin(pitch)) — exactly
    sim_world.py's depth_scan direction formula, confirming this uses the
    identical yaw/pitch sign convention rather than an independently
    invented one.
    """
    fx, fy, cx, cy = _intrinsics_params(intrinsics)
    rot = _CameraRotation.from_pose(camera_pose)

    points: list[Point3] = []
    height, width = depth_mm.shape[:2]
    for v in range(0, height, max(1, stride)):
        for u in range(0, width, max(1, stride)):
            d = float(depth_mm[v, u])
            if not math.isfinite(d) or d <= 0.0 or d > max_range_mm:
                continue

            wx, wy, wz = rot.camera_ray_to_world((u - cx) / fx * d, (v - cy) / fy * d, d)
            points.append((camera_pose.x_mm + wx, camera_pose.y_mm + wy, camera_pose.z_mm + wz))
    return points


def backproject_depth_colorized(
    depth_mm: np.ndarray,
    frame_bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose: CameraPose3D,
    stride: int = 4,
    max_range_mm: float = 3000.0,
) -> list[ColoredPoint3]:
    """Same geometry as backproject_depth, plus the one thing a simulated
    ray-cast has no source for: colour, sampled from `frame_bgr` at each
    backprojected pixel — real hardware's counterpart to sim_world.py's
    fixed SIMULATED_POINT_COLOR placeholder. `frame_bgr` must be the exact
    frame `depth_mm` was estimated from (same resolution as intrinsics'
    pixel grid), since colour is sampled at the identical (u, v) each depth
    reading came from.
    """
    fx, fy, cx, cy = _intrinsics_params(intrinsics)
    rot = _CameraRotation.from_pose(camera_pose)

    points: list[ColoredPoint3] = []
    height, width = depth_mm.shape[:2]
    for v in range(0, height, max(1, stride)):
        for u in range(0, width, max(1, stride)):
            d = float(depth_mm[v, u])
            if not math.isfinite(d) or d <= 0.0 or d > max_range_mm:
                continue

            wx, wy, wz = rot.camera_ray_to_world((u - cx) / fx * d, (v - cy) / fy * d, d)
            b, g, r = frame_bgr[v, u][:3]
            points.append((
                camera_pose.x_mm + wx, camera_pose.y_mm + wy, camera_pose.z_mm + wz,
                int(r), int(g), int(b),
            ))
    return points


def _intrinsics_params(intrinsics: CameraIntrinsics) -> tuple[float, float, float, float]:
    return (
        float(intrinsics.camera_matrix[0, 0]), float(intrinsics.camera_matrix[1, 1]),
        float(intrinsics.camera_matrix[0, 2]), float(intrinsics.camera_matrix[1, 2]),
    )


@dataclass(frozen=True)
class _CameraRotation:
    """The yaw/pitch rotation from backproject_depth's derivation, factored
    out so backproject_depth and backproject_depth_colorized share one
    implementation instead of two copies that could drift apart."""

    cy_: float
    sy_: float
    cp_: float
    sp_: float

    @classmethod
    def from_pose(cls, camera_pose: CameraPose3D) -> "_CameraRotation":
        yaw_r = math.radians(camera_pose.yaw_deg)
        pitch_r = math.radians(camera_pose.pitch_deg)
        return cls(math.cos(yaw_r), math.sin(yaw_r), math.cos(pitch_r), math.sin(pitch_r))

    def camera_ray_to_world(self, x_c: float, y_c: float, z_c: float) -> tuple[float, float, float]:
        a = y_c * self.sp_ + z_c * self.cp_
        wx = a * self.cy_ + x_c * self.sy_
        wy = a * self.sy_ - x_c * self.cy_
        wz = z_c * self.sp_ - y_c * self.cp_
        return wx, wy, wz


def encode_response_depth(depth_mm: np.ndarray) -> bytes:
    """The server side of decode_response_depth's wire format — kept next
    to it so the two can never drift apart independently."""
    depth32 = np.asarray(depth_mm, dtype="<f4")
    height, width = depth32.shape[:2]
    return struct.pack(">II", height, width) + depth32.tobytes()


class DepthEstimator(Protocol):
    """A per-frame depth predictor. Deliberately just this one method, so
    both an on-device model and a remote call can implement it — mission.py
    only ever needs "give me depth for this frame," never which backend
    produced it (same swappability as transport.Transport for Bluetooth)."""

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return an (H, W) depth map in millimetres, one value per pixel of
        `frame_bgr`. Unknown/invalid pixels should be NaN or <= 0 (see
        backproject_depth, which skips both)."""
        ...


def encode_request_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    """Encode a frame for the wire — its own function so
    RemoteDepthEstimator's I/O and its encoding logic can be tested
    separately from any actual socket."""
    import cv2  # imported lazily: only this function needs opencv's encoder

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("failed to JPEG-encode the frame")
    return buf.tobytes()


def decode_response_depth(body: bytes) -> np.ndarray:
    """Parse a depth-server response body: an 8-byte header (big-endian
    uint32 height, uint32 width) followed by height*width little-endian
    float32 values, mm. A small custom binary framing rather than JSON —
    same reasoning as protocol.py's wire format: this is a lot of floats
    (e.g. a 256x256 depth map is 65536 of them), and JSON's per-number text
    overhead is real weight to carry over a Pi's network link for no benefit
    over just... sending the floats."""
    if len(body) < 8:
        raise ValueError(f"response too short to contain a header: {len(body)} bytes")
    height, width = struct.unpack(">II", body[:8])
    expected = 8 + height * width * 4
    if len(body) != expected:
        raise ValueError(
            f"response length {len(body)} doesn't match header-declared "
            f"{height}x{width} depth map ({expected} bytes expected)"
        )
    return np.frombuffer(body[8:], dtype="<f4").reshape(height, width)


class RemoteDepthEstimator:
    """Depth inference offloaded to a bigger machine (a laptop or server
    running depth_server.py) — the Pi 3B has no GPU and 1GB of RAM, nowhere
    near enough to run a depth model itself at any useful speed. The Pi's
    only job here is capturing a frame and making an HTTP call.

    `connection_factory`, if given, replaces the real
    `http.client.HTTPConnection` — this is how tests inject a socketpair
    connection with no real network binding involved (see
    test_depth_estimator.py), the same technique test_server.py uses on the
    server side of this exact same kind of call.
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 10.0,
        connection_factory=None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._connection_factory = connection_factory or (
            lambda: http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        )

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        body = encode_request_jpeg(frame_bgr)
        conn = self._connection_factory()
        try:
            conn.request(
                "POST", "/estimate_depth", body=body,
                headers={"Content-Type": "image/jpeg", "Content-Length": str(len(body))},
            )
            response = conn.getresponse()
            response_body = response.read()
            if response.status != 200:
                raise ValueError(
                    f"depth server returned HTTP {response.status}: "
                    f"{response_body[:200]!r}"
                )
            return decode_response_depth(response_body)
        finally:
            conn.close()


def make_depth_scanner(
    capture_frame: Callable[[], np.ndarray],
    estimator: DepthEstimator,
    intrinsics: CameraIntrinsics,
    geometry: CameraGeometry,
    camera_height_mm: float,
    camera_pitch_deg: float,
    stride: int = 4,
    max_range_mm: float = 3000.0,
) -> Callable[[Telemetry], list[ColoredPoint3]]:
    """Build mission.py's `depth_scanner` hook for real hardware — the
    real-camera counterpart to demo_explore.py's simulated
    make_depth_scanner, composing capture -> estimate -> colorized
    backproject into the exact `Callable[[Telemetry], list[ColoredPoint3]]`
    shape mission.ExplorationMission(depth_scanner=...) expects. Uses
    backproject_depth_colorized (not the plain backproject_depth) since
    real hardware always has the source frame on hand to sample colour
    from — unlike sim_world.py's ray-cast scan, which doesn't.

    `capture_frame` is injected rather than hardcoded to a specific camera
    API (e.g. picamera2) so this composition — the actual logic worth
    testing — is fully testable with a fake frame source and no real camera
    or model, exactly like `estimator` being a DepthEstimator lets tests
    substitute a fake one instead of a real network call or model file (see
    test_depth_estimator.py).
    """

    def scan(telemetry: Telemetry) -> list[ColoredPoint3]:
        frame = capture_frame()
        depth_mm = estimator.estimate(frame)
        chassis_pose = Pose2D(telemetry.x_mm, telemetry.y_mm, telemetry.heading_deg)
        pose = camera_pose_3d(
            chassis_pose, telemetry.turret_deg, geometry, camera_height_mm, camera_pitch_deg
        )
        return backproject_depth_colorized(
            depth_mm, frame, intrinsics, pose, stride=stride, max_range_mm=max_range_mm
        )

    return scan
