"""Tests for vision.py, run against the real opencv/numpy installed on this
machine (not mocked) — this is the one part of the Pi stack that genuinely
needs those libraries, and now that they're installed, there's no reason to
leave it unverified.

What these tests prove: marker generation and detection round-trip
correctly, and the rvec/tvec -> Pose2D conversion is physically correct —
checked by generating synthetic images for known, physically-realistic
camera geometries and confirming the recovered pose, run all the way through
the already-tested localize.py chain, matches ground truth.

What these tests do NOT prove: that a real Pi Camera Module, in a real room,
under real lighting, will detect a real printed marker reliably. That can
only be verified with the actual hardware.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout import vision  # noqa: E402
from scout.localize import (  # noqa: E402
    CameraGeometry,
    chassis_pose_from_marker,
    marker_in_camera_from_chassis,
)
from scout.pose2d import Pose2D, approx_equal  # noqa: E402

CAMERA_MATRIX = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
DIST_COEFFS = np.zeros(5)
INTRINSICS = vision.CameraIntrinsics(camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS)
MARKER_SIZE_MM = 100.0
ZERO_GEOMETRY = CameraGeometry(camera_mount=Pose2D(0, 0, 0), turret_mount=Pose2D(0, 0, 0))


# --------------------------------------------------------------- generation


def test_generate_marker_image_has_the_requested_size():
    img = vision.generate_marker_image(marker_id=3, pixels=200)
    assert img.shape == (200, 200)
    assert img.dtype == np.uint8


def test_generated_markers_are_actually_detectable():
    """Round-trip through real cv2 calls: generate marker 7, paste it into
    a blank canvas, and confirm detection finds exactly that ID."""
    marker_img = vision.generate_marker_image(marker_id=7, pixels=200)
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    canvas[100:300, 200:400] = marker_img

    detector = vision.ArucoDetector(INTRINSICS, MARKER_SIZE_MM)
    detections = detector.detect(canvas)

    assert len(detections) == 1
    assert detections[0].marker_id == 7


def test_detect_returns_empty_list_for_a_blank_image():
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    detector = vision.ArucoDetector(INTRINSICS, MARKER_SIZE_MM)
    assert detector.detect(canvas) == []


def test_different_marker_ids_are_distinguished():
    detector = vision.ArucoDetector(INTRINSICS, MARKER_SIZE_MM)
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    canvas[50:150, 50:150] = vision.generate_marker_image(1, 100)
    canvas[300:400, 450:550] = vision.generate_marker_image(2, 100)

    detections = detector.detect(canvas)
    ids = sorted(d.marker_id for d in detections)
    assert ids == [1, 2]


# --------------------------------------------------------- pose extraction


def _synthesize_detection(pose_in_camera: Pose2D) -> Pose2D:
    """Build a synthetic image containing a marker at exactly the given
    camera-relative pose, run it through the REAL detect+solvePnP pipeline,
    and return the recovered pose — end-to-end through actual cv2 calls,
    not just the conversion function in isolation."""
    detector = vision.ArucoDetector(INTRINSICS, MARKER_SIZE_MM)
    rvec, tvec = vision.pose_to_rvec_tvec(pose_in_camera)

    corners_3d = detector._object_points
    image_points, _ = __import__("cv2").projectPoints(
        corners_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
    )
    marker_img = vision.generate_marker_image(marker_id=0, pixels=300)

    import cv2

    dst = image_points.reshape(4, 2).astype(np.float32)
    src = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    warped = cv2.warpPerspective(
        marker_img, homography, (640, 480), borderValue=255
    )
    mask = cv2.warpPerspective(
        np.full((300, 300), 255, dtype=np.uint8), homography, (640, 480), borderValue=0
    )
    canvas = np.where(mask > 0, warped, canvas)

    detections = detector.detect(canvas)
    assert len(detections) == 1, f"expected exactly one detection, got {len(detections)}"
    return detections[0].pose_in_camera


def test_pose_extraction_round_trips_through_a_real_warped_image():
    """The strongest available test without a physical camera: render an
    actual marker image, perspective-warp it into a synthetic photo exactly
    as a real detection would see it, run REAL detectMarkers + solvePnP,
    and check the recovered pose against what was asked for."""
    original = Pose2D(x_mm=800.0, y_mm=150.0, heading_deg=-150.0)
    recovered = _synthesize_detection(original)
    assert approx_equal(recovered, original, tol_mm=15.0, tol_deg=3.0), (
        f"expected ~{original}, got {recovered}"
    )


def test_end_to_end_localization_through_a_real_rendered_image():
    """The full pipeline: a chassis pose -> what the camera would really
    see (rendered as an actual image and detected) -> back to a chassis
    pose via the tested localize.py chain. This is as close to a real
    hardware test as is possible without a physical camera.

    Marker placed ~1.2m out — close enough for a 100mm marker to have a
    reasonable apparent size (~50px) in a 640x480 synthetic render.

    The tolerance below (40mm / 2deg) is not arbitrary: confirmed by a side
    experiment (bypassing pixel rendering entirely — projectPoints straight
    into solvePnP) that the *exact* math recovers this same scenario to
    within 1e-4mm. So the ~30mm / ~1deg actually observed here is entirely
    pixel-level rasterization and sub-pixel corner-detection noise from
    rendering a ~50px marker, composed through the localization chain,
    where a landmark ~1.2m away amplifies a fraction of a degree of
    camera-frame heading noise into tens of mm of world-position error
    (lever-arm effect — expected in any landmark-based localization, not a
    defect here). A real camera would show the same kind of noise; this is
    optics, not a bug. Worth remembering when placing real markers: prefer
    moderate range over maximum room coverage per marker.
    """
    chassis_world = Pose2D(0.0, 0.0, 0.0)
    marker_world = Pose2D(1200.0, 300.0, -140.0)

    expected_in_camera = marker_in_camera_from_chassis(
        chassis_world, marker_world, 0.0, ZERO_GEOMETRY
    )
    recovered_in_camera = _synthesize_detection(expected_in_camera)
    recovered_chassis = chassis_pose_from_marker(
        marker_world, recovered_in_camera, 0.0, ZERO_GEOMETRY
    )
    assert approx_equal(recovered_chassis, chassis_world, tol_mm=40.0, tol_deg=2.0), (
        f"expected ~{chassis_world}, got {recovered_chassis}"
    )


# ------------------------------------------------------------- intrinsics


def test_intrinsics_save_and_load_round_trip(tmp_path_str):
    path = os.path.join(tmp_path_str, "cam.npz")
    original = vision.CameraIntrinsics(camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS)
    original.save(path)
    loaded = vision.CameraIntrinsics.load(path)
    assert np.allclose(loaded.camera_matrix, CAMERA_MATRIX)
    assert np.allclose(loaded.dist_coeffs, DIST_COEFFS)


# ---------------------------------------------------------------- helpers


def _tmp_dir() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="scout_vision_test_")


if __name__ == "__main__":
    tmp_path_str = _tmp_dir()
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path_str" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                fn(tmp_path_str)
            else:
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
