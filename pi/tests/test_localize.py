"""Tests for the ArUco localization pose chain.

Every test constructs a fully synthetic scenario — a known ground-truth
chassis pose, known fixed geometry, a known marker position — computes what
the camera *would* see via `marker_in_camera_from_chassis` (pure forward
kinematics), feeds that into `chassis_pose_from_marker` (the function real
vision code will call), and checks the original chassis pose comes back out.
No opencv, no camera, no hardware: this proves the math chain is correct
before a single real image is ever processed.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout.localize import (  # noqa: E402
    CameraGeometry,
    chassis_pose_from_marker,
    marker_in_camera_from_chassis,
)
from scout.pose2d import Pose2D, approx_equal  # noqa: E402

# A camera mounted 40mm ahead of the turret axis, facing forward; a turret
# mounted 20mm ahead of the chassis's odometry origin. Representative, not
# measured — real values come from docs/BUILD.md's Phase 3 note once a
# camera is actually mounted.
GEOMETRY = CameraGeometry(
    camera_mount=Pose2D(40.0, 0.0, 0.0),
    turret_mount=Pose2D(20.0, 0.0, 0.0),
)


def assert_round_trip(chassis_world, marker_world, turret_angle_deg, geometry=GEOMETRY):
    observed = marker_in_camera_from_chassis(
        chassis_world, marker_world, turret_angle_deg, geometry
    )
    recovered = chassis_pose_from_marker(marker_world, observed, turret_angle_deg, geometry)
    assert approx_equal(recovered, chassis_world, tol_mm=1e-6, tol_deg=1e-6), (
        f"round trip failed: {chassis_world} -> observed {observed} -> {recovered}"
    )


def test_round_trip_robot_at_origin_facing_marker_head_on():
    assert_round_trip(
        chassis_world=Pose2D(0.0, 0.0, 0.0),
        marker_world=Pose2D(2000.0, 0.0, 180.0),  # 2m ahead, facing back at the robot
        turret_angle_deg=0.0,
    )


def test_round_trip_robot_offset_and_rotated():
    assert_round_trip(
        chassis_world=Pose2D(350.0, -820.0, 47.0),
        marker_world=Pose2D(-1500.0, 900.0, -30.0),
        turret_angle_deg=0.0,
    )


def test_round_trip_with_turret_turned_to_the_side():
    """The marker is off to the robot's side; only visible because the
    turret (not the chassis) is turned toward it."""
    assert_round_trip(
        chassis_world=Pose2D(0.0, 0.0, 0.0),
        marker_world=Pose2D(0.0, 1500.0, -90.0),  # directly to the +y side
        turret_angle_deg=85.0,
    )


def test_round_trip_across_many_random_looking_configurations():
    """A spread of hand-picked (not literally random, so the test stays
    deterministic) configurations spanning quadrants, headings, and turret
    angles — the chain must hold generally, not just in tidy special cases.
    """
    configs = [
        (Pose2D(100, 200, 10), Pose2D(3000, 3000, 225), 0.0),
        (Pose2D(-450, 675, -60), Pose2D(-2000, -500, 90), -45.0),
        (Pose2D(0, 0, 179.0), Pose2D(500, -3000, 0.0), 120.0),
        (Pose2D(1234, -4321, -179.0), Pose2D(-1000, 1000, 135.0), -120.0),
        (Pose2D(-99, -99, 90.0), Pose2D(99, 99, -45.0), 30.0),
    ]
    for chassis_world, marker_world, turret_angle_deg in configs:
        assert_round_trip(chassis_world, marker_world, turret_angle_deg)


def test_marker_dead_ahead_with_no_geometry_offsets_is_a_sanity_checkable_case():
    """A degenerate, fully hand-checkable case: zero camera/turret offsets,
    robot at the origin facing +x, marker 1000mm straight ahead facing back.
    The camera should see the marker exactly 1000mm along its own +x (view)
    axis, dead centre, facing back at it (180 degrees relative)."""
    zero_geometry = CameraGeometry(
        camera_mount=Pose2D(0.0, 0.0, 0.0), turret_mount=Pose2D(0.0, 0.0, 0.0)
    )
    chassis_world = Pose2D(0.0, 0.0, 0.0)
    marker_world = Pose2D(1000.0, 0.0, 180.0)

    observed = marker_in_camera_from_chassis(chassis_world, marker_world, 0.0, zero_geometry)
    assert approx_equal(observed, Pose2D(1000.0, 0.0, 180.0), tol_mm=1e-6)


def test_geometry_offsets_actually_matter():
    """A sanity guard against the geometry composition being silently
    skipped: moving the camera mount must change the observed pose."""
    chassis_world = Pose2D(0.0, 0.0, 0.0)
    marker_world = Pose2D(1000.0, 0.0, 180.0)

    near = marker_in_camera_from_chassis(chassis_world, marker_world, 0.0, GEOMETRY)
    far_geometry = CameraGeometry(
        camera_mount=Pose2D(400.0, 0.0, 0.0), turret_mount=Pose2D(20.0, 0.0, 0.0)
    )
    far = marker_in_camera_from_chassis(chassis_world, marker_world, 0.0, far_geometry)
    assert abs(near.x_mm - far.x_mm) > 1.0


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
