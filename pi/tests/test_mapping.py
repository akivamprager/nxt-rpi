"""Tests for the occupancy grid and frontier exploration logic.

No hardware, no numpy, no opencv — pure algorithm correctness on synthetic
grids.
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout import explore  # noqa: E402
from scout.mapping import (  # noqa: E402
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid,
)


def make_grid(width=20, height=20, cell_size_mm=100.0):
    return OccupancyGrid(width=width, height=height, cell_size_mm=cell_size_mm)


def test_new_grid_is_entirely_unknown():
    grid = make_grid()
    assert all(
        grid.state_at(x, y) == UNKNOWN
        for y in range(grid.height)
        for x in range(grid.width)
    )
    assert grid.coverage() == 0.0


def test_world_to_cell_and_back_are_consistent():
    grid = make_grid(cell_size_mm=100.0)
    cx, cy = grid.world_to_cell(250, 450)
    assert (cx, cy) == (2, 4)
    wx, wy = grid.cell_center(cx, cy)
    assert 200 < wx < 300
    assert 400 < wy < 500


def test_single_ray_marks_free_space_and_one_occupied_endpoint():
    grid = make_grid(cell_size_mm=100.0)
    # Sensor at the origin, facing +x, sees an echo at 100 cm (1000mm = cell 10).
    grid.update_ray(x_mm=0, y_mm=0, bearing_deg=0, range_cm=100, has_echo=True)

    assert grid.is_free(2, 0)   # well short of the echo
    assert grid.is_free(8, 0)   # just short of it
    assert grid.is_occupied(10, 0)  # the echo itself
    assert grid.is_unknown(15, 0)   # never swept


def test_no_echo_carves_free_space_but_marks_nothing_occupied():
    """The core correctness rule: no echo means 'unknown beyond here', not
    'clear forever' and not 'wall exactly at max range'."""
    grid = make_grid(cell_size_mm=100.0)
    grid.update_ray(
        x_mm=0, y_mm=0, bearing_deg=0, range_cm=255, has_echo=False, max_range_cm=150
    )
    assert grid.is_free(5, 0)
    # Nothing at the max-range boundary should read as occupied.
    end_cx, _ = grid.world_to_cell(150 * 10.0, 0)
    assert not grid.is_occupied(end_cx, 0)


def test_repeated_readings_reinforce_rather_than_flip_a_cell():
    """A single stray reading shouldn't be able to overturn many consistent
    ones — that's the entire reason log-odds are accumulated instead of the
    grid just storing the last observation."""
    grid = make_grid(cell_size_mm=100.0)
    for _ in range(5):
        grid.update_ray(0, 0, bearing_deg=0, range_cm=100, has_echo=True)
    assert grid.is_occupied(10, 0)

    # One no-echo reading along the same ray must not immediately erase five
    # confident occupied observations.
    grid.update_ray(0, 0, bearing_deg=0, range_cm=255, has_echo=False, max_range_cm=150)
    assert grid.is_occupied(10, 0)


def test_readings_from_multiple_angles_build_a_consistent_map():
    grid = make_grid(width=40, height=40, cell_size_mm=50.0)
    grid.origin_x_mm = -1000.0
    grid.origin_y_mm = -1000.0
    # A robot at the origin sweeping 8 directions, each seeing a wall at 500mm.
    for bearing in range(0, 360, 45):
        grid.update_ray(0, 0, bearing_deg=bearing, range_cm=50, has_echo=True)

    origin_cx, origin_cy = grid.world_to_cell(0, 0)
    assert grid.is_free(origin_cx, origin_cy)

    # A cell just short of one ray's echo point (bearing 0, echo at 500mm)
    # must be free, and the echo cell itself occupied.
    near_cx, near_cy = grid.world_to_cell(300, 0)
    assert grid.is_free(near_cx, near_cy)
    echo_cx, echo_cy = grid.world_to_cell(500, 0)
    assert grid.is_occupied(echo_cx, echo_cy)

    # 8 rays out to 500mm on a 50mm grid necessarily touch more than a
    # handful of cells; this is a sanity floor, not a tuned percentage.
    known_cells = sum(
        1
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.state_at(x, y) != UNKNOWN
    )
    assert known_cells >= 40


def test_cone_closes_the_angular_gap_between_adjacent_sweep_readings():
    """Regression test for a real bug caught via live testing: a single
    infinitely-thin ray per reading left angular gaps between the mission's
    30-degree-apart sweep angles that never got marked free, close to the
    robot. A frontier cell could then end up sitting in that gap essentially
    at the robot's own position — always 'nearest', never resolvable, since
    no single ray ever crossed it. Two readings 30 degrees apart (matching
    DEFAULT_SWEEP_ANGLES in mission.py) must leave no unknown cell in the
    near field between them.
    """
    grid = make_grid(width=60, height=60, cell_size_mm=50.0)
    grid.origin_x_mm = -1500.0
    grid.origin_y_mm = -1500.0

    grid.update_ray(0, 0, bearing_deg=0.0, range_cm=100, has_echo=True)
    grid.update_ray(0, 0, bearing_deg=30.0, range_cm=100, has_echo=True)

    # Check the near field (100-300mm out, well short of the 1000mm echo)
    # at the bisecting angle of 15 degrees, where the old single-ray model
    # left a gap.
    for distance_mm in (100, 150, 200, 250, 300):
        radians = math.radians(15.0)
        x = distance_mm * math.cos(radians)
        y = distance_mm * math.sin(radians)
        cx, cy = grid.world_to_cell(x, y)
        assert grid.is_free(cx, cy), (
            f"gap at {distance_mm}mm along the bisecting angle between two "
            f"30-degree-apart readings — this is exactly the bug that stalled "
            f"the live exploration demo"
        )


def test_cone_occupied_marking_stays_at_the_center_bearing_only():
    """Widening the free-space carving to a cone must not also widen what
    counts as occupied — that stays a single point, or the map would over-
    claim obstacles across the whole cone width."""
    # Generously sized so a reading at exactly 1000mm range lands well
    # inside the array, not right on its boundary.
    grid = make_grid(width=60, height=60, cell_size_mm=50.0)
    grid.origin_x_mm = -1500.0
    grid.origin_y_mm = -1500.0

    grid.update_ray(0, 0, bearing_deg=0.0, range_cm=100, has_echo=True)

    center_cx, center_cy = grid.world_to_cell(1000, 0)
    assert grid.is_occupied(center_cx, center_cy)

    # A point on the cone's edge at the same range must NOT be occupied —
    # only genuinely free (it was swept, nothing was found there).
    radians = math.radians(15.0)
    edge_x, edge_y = 1000 * math.cos(radians), 1000 * math.sin(radians)
    edge_cx, edge_cy = grid.world_to_cell(edge_x, edge_y)
    assert not grid.is_occupied(edge_cx, edge_cy)


def test_grid_edges_do_not_crash_on_an_out_of_range_ray():
    grid = make_grid(width=5, height=5, cell_size_mm=100.0)
    # A ray that runs straight off the edge of a tiny grid must not raise.
    grid.update_ray(0, 0, bearing_deg=0, range_cm=200, has_echo=True)
    assert True  # reaching here without an exception is the assertion


# ---------------------------------------------------------------- exploration


def test_find_frontier_cells_on_a_simple_map():
    grid = make_grid(width=10, height=10, cell_size_mm=100.0)
    # Carve out a free 3x3 patch in the middle of an otherwise unknown grid.
    for x in range(3, 6):
        for y in range(3, 6):
            grid.log_odds[y][x] = -2.0  # confidently free

    frontier = explore.find_frontier_cells(grid)
    # The ring of free cells touching the unknown exterior should all be frontier.
    assert (3, 3) in frontier  # a corner of the free patch touches unknown
    assert (4, 4) not in frontier  # fully interior free cell, all neighbours free


def test_frontiers_are_empty_when_nothing_borders_unknown():
    grid = make_grid(width=4, height=4, cell_size_mm=100.0)
    for row in grid.log_odds:
        for x in range(len(row)):
            row[x] = -2.0  # the entire grid is free: no unknown to border
    assert explore.find_frontier_cells(grid) == set()


def test_clustering_groups_adjacent_frontier_cells():
    grid = make_grid(width=10, height=10, cell_size_mm=100.0)
    for x in range(2, 6):
        grid.log_odds[5][x] = -2.0  # a free strip bordering unknown above and below

    frontiers = explore.find_frontiers(grid, min_cluster_size=1)
    # All of it is one connected frontier strip, not four separate ones.
    assert len(frontiers) == 1
    assert frontiers[0].size == 4


def test_small_clusters_are_dropped_as_noise():
    grid = make_grid(width=10, height=10, cell_size_mm=100.0)
    grid.log_odds[5][5] = -2.0  # a single isolated free cell
    frontiers = explore.find_frontiers(grid, min_cluster_size=2)
    assert frontiers == []


def test_nearest_frontier_picks_the_closest_by_straight_line_distance():
    grid = make_grid(width=20, height=20, cell_size_mm=100.0)
    for x in range(2, 4):
        grid.log_odds[2][x] = -2.0   # near cluster
    for x in range(15, 18):
        grid.log_odds[15][x] = -2.0  # far cluster

    frontiers = explore.find_frontiers(grid, min_cluster_size=1)
    assert len(frontiers) == 2
    chosen = explore.nearest_frontier(frontiers, x_mm=250, y_mm=250)
    assert chosen is not None
    assert chosen.centroid_y_mm < 1000  # picked the near cluster, not the far one


def test_nearest_frontier_on_empty_list_is_none():
    assert explore.nearest_frontier([], 0, 0) is None


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}")
            failed += 1
        else:
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
