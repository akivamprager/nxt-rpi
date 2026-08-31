"""Tests for config.py's validation logic.

Deliberately tests `from_dict` only, with plain Python dicts as fixtures —
this is the part with actual logic, and it needs no YAML library to test.
`load()`'s file-reading wrapper is a couple of lines around `yaml.safe_load`
and is left unverified until PyYAML is installed, same as the rest of
Phase 3's opencv dependency.
"""

import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout import config as cfg  # noqa: E402
from scout.pose2d import Pose2D  # noqa: E402

VALID = {
    "robot": {
        "wheel_diameter_mm": 56.0,
        "track_width_mm": 115.0,
        "turret_gear_ratio": 1.0,
        "turret_max_angle_deg": 120.0,
    },
    "camera": {
        "camera_mount": {"x_mm": 40.0, "y_mm": 0.0, "heading_deg": 0.0},
        "turret_mount": {"x_mm": 20.0, "y_mm": 0.0, "heading_deg": 0.0},
        "intrinsics_path": "calibration/camera.npz",
        "camera_height_mm": 150.0,
        "camera_pitch_deg": -20.0,
        "marker_size_mm": 100.0,
    },
    "grid": {
        "width": 60,
        "height": 60,
        "cell_size_mm": 100.0,
        "origin_x_mm": -3000.0,
        "origin_y_mm": -3000.0,
    },
    "mission": {
        "sweep_angles_deg": [-90, -60, -30, 0, 30, 60, 90],
        "travel_step_mm": 400.0,
        "min_frontier_cluster": 3,
    },
    "markers": [
        {"id": 0, "pose": {"x_mm": 3000.0, "y_mm": 0.0, "heading_deg": 180.0}},
        {"id": 1, "pose": {"x_mm": 0.0, "y_mm": 3000.0, "heading_deg": -90.0}},
    ],
}


def test_valid_config_builds_successfully():
    result = cfg.from_dict(VALID)
    assert result.robot.wheel_diameter_mm == 56.0
    assert result.camera.camera_mount == Pose2D(40.0, 0.0, 0.0)
    assert result.camera.camera_height_mm == 150.0
    assert result.camera.camera_pitch_deg == -20.0
    assert result.camera.marker_size_mm == 100.0
    assert result.grid.width == 60
    assert result.mission.sweep_angles_deg == (-90, -60, -30, 0, 30, 60, 90)
    assert result.markers[0] == Pose2D(3000.0, 0.0, 180.0)
    assert result.markers[1] == Pose2D(0.0, 3000.0, -90.0)


def test_markers_are_optional():
    data = copy.deepcopy(VALID)
    del data["markers"]
    result = cfg.from_dict(data)
    assert result.markers == {}


def test_missing_top_level_section_is_rejected():
    for section in ("robot", "camera", "grid", "mission"):
        data = copy.deepcopy(VALID)
        del data[section]
        try:
            cfg.from_dict(data)
        except cfg.ConfigError:
            continue
        raise AssertionError(f"missing '{section}' section should have been rejected")


def test_missing_field_within_a_section_is_rejected():
    data = copy.deepcopy(VALID)
    del data["robot"]["wheel_diameter_mm"]
    try:
        cfg.from_dict(data)
    except cfg.ConfigError as exc:
        assert "wheel_diameter_mm" in str(exc)
    else:
        raise AssertionError("expected a ConfigError")


def test_wrong_type_is_rejected_not_silently_coerced():
    """A string where a number belongs is a real mistake (e.g. quoting a
    measurement in the YAML by accident) and must fail loudly, not become
    0.0 or crash somewhere far from the actual error."""
    data = copy.deepcopy(VALID)
    data["robot"]["wheel_diameter_mm"] = "56"
    try:
        cfg.from_dict(data)
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("a string measurement should have been rejected")


def test_booleans_are_not_accepted_as_numbers():
    """bool is a subclass of int in Python — worth guarding explicitly so a
    stray `true`/`false` typo doesn't silently become 1.0/0.0."""
    data = copy.deepcopy(VALID)
    data["robot"]["turret_gear_ratio"] = True
    try:
        cfg.from_dict(data)
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("a boolean should not be accepted as a measurement")


def test_duplicate_marker_ids_are_rejected():
    data = copy.deepcopy(VALID)
    data["markers"].append(
        {"id": 0, "pose": {"x_mm": 1.0, "y_mm": 1.0, "heading_deg": 0.0}}
    )
    try:
        cfg.from_dict(data)
    except cfg.ConfigError as exc:
        assert "0" in str(exc)
    else:
        raise AssertionError("a duplicate marker id should have been rejected")


def test_empty_sweep_angles_is_rejected():
    data = copy.deepcopy(VALID)
    data["mission"]["sweep_angles_deg"] = []
    try:
        cfg.from_dict(data)
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("an empty sweep angle list should have been rejected")


def test_non_mapping_top_level_is_rejected():
    try:
        cfg.from_dict(["not", "a", "mapping"])
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("a non-dict top level should have been rejected")


def test_malformed_pose_is_rejected():
    data = copy.deepcopy(VALID)
    data["camera"]["camera_mount"] = [40.0, 0.0, 0.0]  # a list, not a mapping
    try:
        cfg.from_dict(data)
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("a list in place of a pose mapping should have been rejected")


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
