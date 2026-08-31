"""Tests for the 3D point cloud accumulator's dedup/rounding/cap logic.

No hardware, no numpy — pure algorithm correctness on synthetic points.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout.pointcloud import PointCloudMap  # noqa: E402


def test_empty_map_has_zero_points():
    pc = PointCloudMap()
    assert len(pc) == 0
    assert pc.to_dict()["points"] == []


def test_a_single_point_is_stored():
    pc = PointCloudMap(resolution_mm=10.0)
    pc.add_points([(100.0, 200.0, 300.0)])
    assert len(pc) == 1


def test_points_within_the_same_resolution_cell_are_deduplicated():
    """This is the whole point of rounding: two nearby scan hits from
    different sweeps of the same real surface should collapse to one
    stored point, not grow the cloud forever as the robot keeps scanning."""
    pc = PointCloudMap(resolution_mm=20.0)
    pc.add_points([(100.0, 100.0, 100.0)])
    pc.add_points([(105.0, 98.0, 103.0)])  # well within one 20mm cell
    assert len(pc) == 1


def test_points_in_different_cells_are_kept_separately():
    pc = PointCloudMap(resolution_mm=20.0)
    pc.add_points([(0.0, 0.0, 0.0), (500.0, 0.0, 0.0)])
    assert len(pc) == 2


def test_repeated_scans_of_a_static_surface_do_not_grow_unboundedly():
    """The realistic case this exists for: the same wall, scanned from the
    same spot, many sweep cycles in a row."""
    pc = PointCloudMap(resolution_mm=20.0)
    surface = [(300.0, float(y), 100.0) for y in range(-500, 500, 50)]
    for _ in range(20):
        pc.add_points(surface)
    assert len(pc) == len(surface)


def test_max_points_caps_growth():
    pc = PointCloudMap(resolution_mm=1.0, max_points=10)
    # Each point is in its own 1mm cell, so nothing here dedups away.
    pc.add_points([(float(i), 0.0, 0.0) for i in range(50)])
    assert len(pc) == 10


def test_clear_empties_the_map():
    pc = PointCloudMap()  # default 20mm resolution
    pc.add_points([(0.0, 0.0, 0.0), (500.0, 0.0, 0.0)])  # clearly separate cells
    assert len(pc) == 2
    pc.clear()
    assert len(pc) == 0


def test_to_dict_round_trips_the_resolution_and_point_values():
    pc = PointCloudMap(resolution_mm=10.0)
    pc.add_points([(123.0, 45.0, 67.0)])
    data = pc.to_dict()
    assert data["resolution_mm"] == 10.0
    assert len(data["points"]) == 1
    x, y, z = data["points"][0]
    # Rounded to the nearest 10mm.
    assert x == round(123.0 / 10.0) * 10.0
    assert y == round(45.0 / 10.0) * 10.0
    assert z == round(67.0 / 10.0) * 10.0


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
