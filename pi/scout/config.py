"""Loads and validates config.yaml: every number that's specific to this
physical robot build, in one place, so plugging in real hardware means
filling in measurements rather than writing code.

Split deliberately into two layers:

- `from_dict` builds and validates a `Config` from a plain dict, with no I/O
  at all. This is where the actual logic lives, and it's fully testable
  right now with plain Python dicts as fixtures — no YAML library needed.
- `load` is a thin wrapper that reads a file and hands `yaml.safe_load`'s
  result to `from_dict`. This is the only part that touches PyYAML (see
  requirements.txt), and there is deliberately almost nothing in it to get
  wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pose2d import Pose2D


class ConfigError(ValueError):
    """The config dict is missing a key or has a value of the wrong shape."""


@dataclass(frozen=True)
class RobotGeometry:
    """Measured once, per BUILD.md's "before Phase 1" and "note for later
    (Phase 3)" sections. Every downstream pose estimate inherits these."""

    wheel_diameter_mm: float
    track_width_mm: float
    turret_gear_ratio: float
    turret_max_angle_deg: float


@dataclass(frozen=True)
class CameraGeometryConfig:
    """The camera/turret offsets consumed by localize.CameraGeometry, plus
    the calibration file path — kept separate from localize.CameraGeometry
    itself so this module has no import-time dependency on how vision.py
    eventually structures its calibration data.

    `camera_height_mm`/`camera_pitch_deg` are the two measurements
    depth_estimator.camera_pose_3d needs beyond what CameraGeometry already
    has: camera_mount/turret_mount are 2D-only (Pose2D has no z or tilt),
    because the turret itself only ever yaws — there is no second axis of
    motion (see the plan's "three servos total" constraint). A fixed
    mounting height and downward tilt are enough to place the camera in 3D
    given that one constraint."""

    camera_mount: Pose2D
    turret_mount: Pose2D
    intrinsics_path: str
    camera_height_mm: float
    camera_pitch_deg: float
    #: The physical size (edge length) of printed ArUco markers, mm — feeds
    #: vision.ArucoDetector's solvePnP call directly. Every marker in this
    #: build must be printed at this same size (see docs/BUILD.md).
    marker_size_mm: float


@dataclass(frozen=True)
class GridConfig:
    width: int
    height: int
    cell_size_mm: float
    origin_x_mm: float
    origin_y_mm: float


@dataclass(frozen=True)
class MissionConfig:
    sweep_angles_deg: tuple[float, ...]
    travel_step_mm: float
    min_frontier_cluster: int


@dataclass(frozen=True)
class Config:
    robot: RobotGeometry
    camera: CameraGeometryConfig
    grid: GridConfig
    mission: MissionConfig
    #: marker id -> its surveyed world pose
    markers: dict[int, Pose2D] = field(default_factory=dict)


def _require(data: dict, key: str, path: str) -> Any:
    if key not in data:
        raise ConfigError(f"{path}.{key} is missing")
    return data[key]


def _as_float(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{path} must be a number, got {value!r}")
    return float(value)


def _as_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path} must be an integer, got {value!r}")
    return value


def _as_pose(data: Any, path: str) -> Pose2D:
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping with x_mm/y_mm/heading_deg")
    return Pose2D(
        x_mm=_as_float(_require(data, "x_mm", path), f"{path}.x_mm"),
        y_mm=_as_float(_require(data, "y_mm", path), f"{path}.y_mm"),
        heading_deg=_as_float(_require(data, "heading_deg", path), f"{path}.heading_deg"),
    )


def _build_robot(data: dict) -> RobotGeometry:
    section = _require(data, "robot", "config")
    return RobotGeometry(
        wheel_diameter_mm=_as_float(
            _require(section, "wheel_diameter_mm", "robot"), "robot.wheel_diameter_mm"
        ),
        track_width_mm=_as_float(
            _require(section, "track_width_mm", "robot"), "robot.track_width_mm"
        ),
        turret_gear_ratio=_as_float(
            _require(section, "turret_gear_ratio", "robot"), "robot.turret_gear_ratio"
        ),
        turret_max_angle_deg=_as_float(
            _require(section, "turret_max_angle_deg", "robot"),
            "robot.turret_max_angle_deg",
        ),
    )


def _build_camera(data: dict) -> CameraGeometryConfig:
    section = _require(data, "camera", "config")
    return CameraGeometryConfig(
        camera_mount=_as_pose(_require(section, "camera_mount", "camera"), "camera.camera_mount"),
        turret_mount=_as_pose(_require(section, "turret_mount", "camera"), "camera.turret_mount"),
        intrinsics_path=str(_require(section, "intrinsics_path", "camera")),
        camera_height_mm=_as_float(
            _require(section, "camera_height_mm", "camera"), "camera.camera_height_mm"
        ),
        camera_pitch_deg=_as_float(
            _require(section, "camera_pitch_deg", "camera"), "camera.camera_pitch_deg"
        ),
        marker_size_mm=_as_float(
            _require(section, "marker_size_mm", "camera"), "camera.marker_size_mm"
        ),
    )


def _build_grid(data: dict) -> GridConfig:
    section = _require(data, "grid", "config")
    return GridConfig(
        width=_as_int(_require(section, "width", "grid"), "grid.width"),
        height=_as_int(_require(section, "height", "grid"), "grid.height"),
        cell_size_mm=_as_float(_require(section, "cell_size_mm", "grid"), "grid.cell_size_mm"),
        origin_x_mm=_as_float(_require(section, "origin_x_mm", "grid"), "grid.origin_x_mm"),
        origin_y_mm=_as_float(_require(section, "origin_y_mm", "grid"), "grid.origin_y_mm"),
    )


def _build_mission(data: dict) -> MissionConfig:
    section = _require(data, "mission", "config")
    angles = _require(section, "sweep_angles_deg", "mission")
    if not isinstance(angles, list) or not angles:
        raise ConfigError("mission.sweep_angles_deg must be a non-empty list")
    return MissionConfig(
        sweep_angles_deg=tuple(
            _as_float(a, f"mission.sweep_angles_deg[{i}]") for i, a in enumerate(angles)
        ),
        travel_step_mm=_as_float(
            _require(section, "travel_step_mm", "mission"), "mission.travel_step_mm"
        ),
        min_frontier_cluster=_as_int(
            _require(section, "min_frontier_cluster", "mission"),
            "mission.min_frontier_cluster",
        ),
    )


def _build_markers(data: dict) -> dict[int, Pose2D]:
    section = data.get("markers", [])
    if not isinstance(section, list):
        raise ConfigError("markers must be a list")
    markers: dict[int, Pose2D] = {}
    for i, entry in enumerate(section):
        marker_id = _as_int(_require(entry, "id", f"markers[{i}]"), f"markers[{i}].id")
        if marker_id in markers:
            raise ConfigError(f"marker id {marker_id} is defined more than once")
        markers[marker_id] = _as_pose(
            _require(entry, "pose", f"markers[{i}]"), f"markers[{i}].pose"
        )
    return markers


def from_dict(data: dict) -> Config:
    """Build and validate a Config from a plain dict — the shape you'd get
    from `yaml.safe_load` or `json.load`. No I/O, so this is what the tests
    exercise directly."""
    if not isinstance(data, dict):
        raise ConfigError("config must be a mapping at the top level")
    return Config(
        robot=_build_robot(data),
        camera=_build_camera(data),
        grid=_build_grid(data),
        mission=_build_mission(data),
        markers=_build_markers(data),
    )


def load(path: str) -> Config:
    """Read and validate config.yaml from disk. Needs PyYAML installed."""
    import yaml  # imported lazily: only this function touches it

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return from_dict(data or {})
