"""ArUco marker detection and camera calibration.

This is the hardware-dependent layer of Phase 3, but only barely: the actual
math (rvec/tvec -> Pose2D) is validated against real `cv2.projectPoints` /
`cv2.solvePnP` calls on physically-realistic synthetic scenarios, round-
tripped through the already-tested `localize.py` chain and checked against
ground truth — see test_vision.py. Marker detection and generation are also
exercised against real `cv2.aruco` calls on synthetic images.

What remains genuinely unverified, and can only be verified once a Pi Camera
Module exists, is real-world image quality: focus, exposure, motion blur,
and how reliably `detectMarkers` finds a physical printed marker under real
lighting. That's a camera-quality question, not a math question, and no
amount of synthetic testing substitutes for it.

Uses the modern cv2.aruco API (`getPredefinedDictionary` + `ArucoDetector`),
not the older `Dictionary_get` / `estimatePoseSingleMarkers`, which are
removed in OpenCV >= 4.7 — confirmed against the actually-installed version
(see this session's interactive checks) rather than assumed from memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .pose2d import Pose2D

DEFAULT_DICTIONARY = cv2.aruco.DICT_4X4_50


@dataclass(frozen=True)
class CameraIntrinsics:
    """A calibrated camera's intrinsic matrix and distortion coefficients.

    Produced by `calibrate_from_chessboard_images` (or LEGO... no, a real
    checkerboard) once, stored at config.yaml's `camera.intrinsics_path`,
    and loaded once at startup.
    """

    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (5,) or (8,), per cv2.calibrateCamera's output

    def save(self, path: str) -> None:
        np.savez(path, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)

    @classmethod
    def load(cls, path: str) -> "CameraIntrinsics":
        data = np.load(path)
        return cls(camera_matrix=data["camera_matrix"], dist_coeffs=data["dist_coeffs"])


@dataclass(frozen=True)
class Detection:
    marker_id: int
    #: The marker's pose relative to the camera, projected onto the floor
    #: plane — feed this straight into localize.chassis_pose_from_marker
    #: along with the marker's known world pose from config.yaml.
    pose_in_camera: Pose2D


def pose_to_rvec_tvec(pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of `_rvec_tvec_to_pose2d` — mainly useful for generating
    synthetic test images (see test_vision.py), but exposed publicly since
    it's the natural counterpart to `_rvec_tvec_to_pose2d` and may be useful
    for a future camera simulator paralleling sim_world.py."""
    tvec = np.array([-pose.y_mm, 0.0, pose.x_mm])
    rvec, _ = cv2.Rodrigues(np.array([0.0, math.radians(-pose.heading_deg), 0.0]))
    return rvec, tvec


def _rvec_tvec_to_pose2d(rvec: np.ndarray, tvec: np.ndarray) -> Pose2D:
    """Marker-relative-to-camera (rvec, tvec) from solvePnP, collapsed to a
    floor-plane Pose2D.

    Assumes the camera is mounted roughly level and the marker is mounted
    roughly upright — both true by construction per docs/BUILD.md — so the
    camera's vertical (Y) axis can be ignored and only the X-Z (floor) plane
    matters.

    Axis mapping (OpenCV convention: X right, Y down, Z forward):
      - marker's forward distance  -> tvec.z
      - marker's left/right offset -> -tvec.x  (camera +X is right, so left
        is negative X; this project's +y is left, matching Pose2D/world
        heading convention where positive is counter-clockwise)
      - marker's facing angle      -> the floor-projected direction of the
        marker's own local +Z axis (its outward normal — R's third column),
        negated because a positive rotation about camera Y is clockwise
        from a bird's-eye view, but this project's heading convention is
        counter-clockwise-positive (matches ScoutServer's doDrive: "omega,
        positive counter-clockwise").

    All three signs were derived empirically against real cv2 calls, not
    assumed from documentation — see this session's interactive checks and
    test_vision.py's round-trip tests through the tested localize.py chain.
    """
    rotation, _ = cv2.Rodrigues(rvec)
    tvec = tvec.ravel()
    normal = rotation[:, 2]
    heading_deg = -math.degrees(math.atan2(normal[0], normal[2]))
    return Pose2D(x_mm=float(tvec[2]), y_mm=float(-tvec[0]), heading_deg=heading_deg)


class ArucoDetector:
    """Finds ArUco markers in a frame and returns their pose relative to
    the camera."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        marker_size_mm: float,
        dictionary_id: int = DEFAULT_DICTIONARY,
    ) -> None:
        self.intrinsics = intrinsics
        self.marker_size_mm = marker_size_mm
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._detector = cv2.aruco.ArucoDetector(
            self._dictionary, cv2.aruco.DetectorParameters()
        )
        half = marker_size_mm / 2.0
        # Object points in the marker's own local frame (Z=0 plane), in the
        # same corner order detectMarkers returns: top-left, top-right,
        # bottom-right, bottom-left.
        self._object_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )

    def detect(self, frame_gray: np.ndarray) -> list[Detection]:
        """`frame_gray` must be a single-channel (grayscale) uint8 image.

        The camera capture loop is responsible for the color/format
        conversion — see the note in capture.py about what picamera2 hands
        back.
        """
        corners, ids, _rejected = self._detector.detectMarkers(frame_gray)
        if ids is None:
            return []

        detections = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points,
                marker_corners.reshape(-1, 2),
                self.intrinsics.camera_matrix,
                self.intrinsics.dist_coeffs,
            )
            if not ok:
                continue
            detections.append(
                Detection(
                    marker_id=int(marker_id),
                    pose_in_camera=_rvec_tvec_to_pose2d(rvec, tvec),
                )
            )
        return detections


def generate_marker_image(
    marker_id: int, pixels: int, dictionary_id: int = DEFAULT_DICTIONARY
) -> np.ndarray:
    """A printable marker image — see markers/ and docs/BUILD.md."""
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)


def calibrate_from_chessboard_images(
    images: list[np.ndarray],
    board_size: tuple[int, int],
    square_size_mm: float,
) -> CameraIntrinsics:
    """Solve for camera intrinsics from a set of chessboard photos.

    `board_size` is the count of INTERIOR corners (columns, rows) — for a
    9x7-square board, that's (8, 6). Standard OpenCV calibration recipe;
    correct per the documented API, but **unverified against real images**,
    since no camera exists yet to photograph an actual checkerboard with.
    Verify this once real calibration photos exist, the same way
    ProtocolTest.java could only be verified once a JDK existed.

    Raises ValueError if fewer than 3 images yield detectable corners —
    calibration accuracy needs many views from different angles; the
    OpenCV/picamera2 tutorials suggest 15-20 as a practical target.
    """
    object_points_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_points_template[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(
        -1, 2
    )
    object_points_template *= square_size_mm

    object_points = []
    image_points = []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image in images:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size)
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_points_template)
        image_points.append(refined)

    if len(object_points) < 3:
        raise ValueError(
            f"only {len(object_points)} of {len(images)} images had a detectable "
            f"{board_size} chessboard; need at least 3, ideally 15-20 from "
            f"varied angles and distances"
        )

    _rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return CameraIntrinsics(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
