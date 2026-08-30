"""ArUco marker-based localization: the pose chain from "marker as seen from
the camera" to "robot's corrected world pose."

This is the part of Phase 3 that has nothing to do with opencv. Detecting a
marker in a camera frame and running solvePnP gives a marker pose relative
to the camera; turning that into a corrected robot pose is pure 2D rigid-
transform algebra on top of pose2d.py, and is fully buildable and testable
without a camera, a Pi, or an NXT — see test_localize.py for round-trip
proofs against synthetic scenarios.

vision.py is the (currently unverified, opencv-dependent) thin layer that
will call `marker_in_camera` here with real solvePnP output; everything
below it is already proven correct.

Frame chain
-----------
::

    world --[marker_world]--> marker
    marker --[inverse(marker_in_camera)]--> camera   (marker_in_camera: how
                                                        the marker looks FROM
                                                        the camera)
    camera --[inverse(camera_mount)]--> turret axis   (camera_mount: fixed,
                                                        measured once)
    turret axis --[inverse(turret_pose(angle))]--> chassis

Each `-->` is a `pose2d.compose`; composing world_T_marker with the chain
above and stripping the final inversion yields world_T_chassis, exactly
mirroring how `SET_POSE` then overwrites the firmware's odometry with that
result (see robot.Robot.set_pose and ScoutServer's CMD_SET_POSE handler).
"""

from __future__ import annotations

from dataclasses import dataclass

from .pose2d import Pose2D, compose, inverse


@dataclass(frozen=True)
class CameraGeometry:
    """Fixed, one-time-measured offsets for this specific physical build.

    Both offsets are in the mounting surface's own local frame: `camera_mount`
    is the camera's pose relative to the turret's rotation axis (measured
    with the turret at its zero/centre position), and `turret_mount` is the
    turret axis's position on the chassis, relative to the chassis's
    odometry origin. See docs/BUILD.md's "note for later (Phase 3)" — these
    are the two measurements to take once the camera is physically mounted.
    """

    camera_mount: Pose2D
    turret_mount: Pose2D


def chassis_pose_from_marker(
    marker_world: Pose2D,
    marker_in_camera: Pose2D,
    turret_angle_deg: float,
    geometry: CameraGeometry,
) -> Pose2D:
    """The robot's corrected world pose, from one marker sighting.

    Args:
        marker_world: the marker's known, surveyed pose in the world frame
            (position plus the direction it faces outward from its wall).
        marker_in_camera: the marker's pose as seen from the camera — a 2D
            simplification of solvePnP's (rvec, tvec): how far ahead and to
            the side the marker is, and its rotation relative to the
            camera's own view direction, projected onto the floor plane.
        turret_angle_deg: the turret's current bearing relative to the
            chassis (matches Telemetry.turret_deg).
        geometry: this build's fixed camera/turret offsets.

    Returns:
        The chassis's corrected pose in the world/odometry frame — feed
        this straight into Robot.set_pose() to correct drift.
    """
    world_camera = compose(marker_world, inverse(marker_in_camera))
    world_turret = compose(world_camera, inverse(geometry.camera_mount))

    turret_pose = Pose2D(
        geometry.turret_mount.x_mm, geometry.turret_mount.y_mm, turret_angle_deg
    )
    world_chassis = compose(world_turret, inverse(turret_pose))
    return world_chassis.normalized()


def marker_in_camera_from_chassis(
    chassis_world: Pose2D,
    marker_world: Pose2D,
    turret_angle_deg: float,
    geometry: CameraGeometry,
) -> Pose2D:
    """The forward direction of the chain above: given a true chassis pose,
    what would the camera actually observe?

    This exists for testing — it lets test_localize.py construct a fully
    synthetic, internally-consistent scenario (pick a chassis pose, compute
    what the camera would see, feed that into `chassis_pose_from_marker`,
    and assert you get the original chassis pose back) without needing any
    real image or opencv call at all.
    """
    turret_pose = Pose2D(
        geometry.turret_mount.x_mm, geometry.turret_mount.y_mm, turret_angle_deg
    )
    world_turret = compose(chassis_world, turret_pose)
    world_camera = compose(world_turret, geometry.camera_mount)
    camera_marker = compose(inverse(world_camera), marker_world)
    return camera_marker.normalized()
