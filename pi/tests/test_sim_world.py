"""Geometry correctness tests for sim_world.py.

Ray-segment intersection is exactly the kind of code that looks right and
is subtly wrong (sign errors, off-by-one on the inclusive/exclusive bounds,
parallel-ray edge cases). Everything here is checked against hand-computed
expected values, not just "does it run."
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from sim_world import (  # noqa: E402
    SIMULATED_FLOOR_COLOR,
    SIMULATED_WALL_COLOR,
    FurnitureBox,
    SimulatedRoom,
    _ray_box_distance,
)


def test_sense_hits_wall_straight_ahead():
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))])
    range_cm, has_echo = room.sense(0, 0, bearing_deg=0)
    assert has_echo
    assert range_cm == 10  # 100 mm == 10 cm


def test_sense_no_echo_when_nothing_in_range():
    room = SimulatedRoom(walls=[((5000.0, -1000.0), (5000.0, 1000.0))])
    range_cm, has_echo = room.sense(0, 0, bearing_deg=0, max_range_cm=255)
    assert not has_echo


def test_sense_ignores_walls_behind_the_ray():
    """A wall behind the sensor must not register — t must be >= 0."""
    room = SimulatedRoom(walls=[((-100.0, -1000.0), (-100.0, 1000.0))])
    range_cm, has_echo = room.sense(0, 0, bearing_deg=0)
    assert not has_echo


def test_sense_picks_the_nearest_of_several_walls():
    room = SimulatedRoom(
        walls=[
            ((300.0, -1000.0), (300.0, 1000.0)),
            ((150.0, -1000.0), (150.0, 1000.0)),
            ((500.0, -1000.0), (500.0, 1000.0)),
        ]
    )
    range_cm, has_echo = room.sense(0, 0, bearing_deg=0)
    assert has_echo
    assert range_cm == 15


def test_sense_respects_bearing_not_just_distance():
    """A closer wall in the wrong direction must not be hit."""
    room = SimulatedRoom(
        walls=[
            ((50.0, -1000.0), (50.0, 1000.0)),   # dead ahead at bearing 0, close
        ]
    )
    # Facing directly away (180 deg) must not see the wall that's behind it.
    range_cm, has_echo = room.sense(0, 0, bearing_deg=180)
    assert not has_echo


def test_sense_at_an_angle():
    """A wall hit at 45 degrees: verify against hand-computed geometry.

    Ray from origin at 45deg is the line y = x. A vertical wall at x=100
    (from y=-1000 to 1000) is hit at (100, 100), distance = 100*sqrt(2).
    """
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))])
    range_cm, has_echo = room.sense(0, 0, bearing_deg=45)
    assert has_echo
    expected_cm = round(100 * math.sqrt(2) / 10.0)
    assert abs(range_cm - expected_cm) <= 1


def test_sense_misses_a_wall_that_ends_before_the_ray_crosses_its_line():
    """The infinite line the wall lies on is hit, but the finite segment
    doesn't reach that far — must not register (u must stay in [0, 1])."""
    room = SimulatedRoom(walls=[((100.0, 50.0), (100.0, 200.0))])  # doesn't cross y=0
    range_cm, has_echo = room.sense(0, 0, bearing_deg=0)
    assert not has_echo


def test_rectangle_room_hits_the_correct_wall_from_center():
    room = SimulatedRoom.rectangle(2000, 1000)
    cx, cy = 1000.0, 500.0  # dead center

    range_cm, has_echo = room.sense(cx, cy, bearing_deg=0)  # facing +x (right wall)
    assert has_echo and abs(range_cm - 100) <= 1  # 1000mm to the right wall

    range_cm, has_echo = room.sense(cx, cy, bearing_deg=90)  # facing +y (top wall)
    assert has_echo and abs(range_cm - 50) <= 1  # 500mm to the top wall


def test_clearance_is_zero_on_a_wall_and_positive_away_from_it():
    room = SimulatedRoom(walls=[((0.0, 0.0), (1000.0, 0.0))])
    assert room.clearance(500, 0) < 1.0
    assert room.clearance(500, 200) > 199.0


def test_clearance_uses_nearest_point_on_segment_not_the_infinite_line():
    """Near the wall's own line but past its endpoint: distance is to the
    endpoint, not to the (nonexistent) continuation of the line."""
    room = SimulatedRoom(walls=[((0.0, 0.0), (100.0, 0.0))])
    # (200, 0) is past the segment's end at (100, 0): nearest point is (100,0).
    assert abs(room.clearance(200, 0) - 100.0) < 1e-6


def test_is_colliding_respects_the_robot_radius():
    room = SimulatedRoom(walls=[((0.0, 0.0), (1000.0, 0.0))])
    assert room.is_colliding(500, 50, robot_radius_mm=90)
    assert not room.is_colliding(500, 200, robot_radius_mm=90)


def test_add_wall_creates_an_obstacle_the_ray_can_hit():
    room = SimulatedRoom.rectangle(2000, 2000)
    room.add_wall((900.0, 0.0), (900.0, 1100.0))  # an interior divider
    range_cm, has_echo = room.sense(500, 500, bearing_deg=0)
    assert has_echo
    assert abs(range_cm - 40) <= 1  # 400mm to the divider, not 1500mm to the far wall


# ------------------------------------------------------------- _cast_3d


def test_level_ray_hits_the_wall_at_floor_height():
    """A ray with no vertical component never leaves z=0, so it must hit the
    wall at its very base — height 0, not the floor (dz=0 means the floor
    branch's dz < 0 check never fires)."""
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))], wall_height_mm=900.0)
    hit = room._cast_3d(0.0, 0.0, 0.0, dx=1.0, dy=0.0, dz=0.0, max_range_mm=1000.0)
    assert hit is not None
    x, y, z, r, g, b = hit
    assert abs(x - 100.0) < 1e-6
    assert abs(y - 0.0) < 1e-6
    assert abs(z - 0.0) < 1e-6
    assert (r, g, b) == SIMULATED_WALL_COLOR


def test_straight_down_ray_hits_the_floor():
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))], wall_height_mm=900.0)
    # pitch -90deg: dx=cos(-90)*cos(yaw)=0, dy=0, dz=sin(-90)=-1 — straight down.
    hit = room._cast_3d(0.0, 0.0, 500.0, dx=0.0, dy=0.0, dz=-1.0, max_range_mm=1000.0)
    assert hit is not None
    x, y, z, r, g, b = hit
    assert abs(x - 0.0) < 1e-6
    assert abs(y - 0.0) < 1e-6
    assert abs(z - 0.0) < 1e-6  # hit the floor exactly below the start point
    assert (r, g, b) == SIMULATED_FLOOR_COLOR


def test_ray_angled_up_over_the_wall_hits_nothing():
    """Steep enough to clear the wall's height before reaching its XY line,
    and pointing up (never crosses the floor either) — a real miss, not a
    computation that should quietly return something anyway."""
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))], wall_height_mm=900.0)
    pitch = math.radians(85.0)  # tan(85deg) ~= 11.4 > 900/100=9, clears the wall
    dx, dz = math.cos(pitch), math.sin(pitch)
    hit = room._cast_3d(0.0, 0.0, 0.0, dx=dx, dy=0.0, dz=dz, max_range_mm=5000.0)
    assert hit is None


def test_downward_angled_ray_hits_the_wall_at_the_hand_computed_height():
    """A genuinely independent check: distance and height computed by hand
    (see this test's derivation in the accompanying commit), not just
    re-run through the same code being tested."""
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))], wall_height_mm=900.0)
    pitch = math.radians(-10.0)
    dx, dz = math.cos(pitch), math.sin(pitch)
    hit = room._cast_3d(0.0, 0.0, 800.0, dx=dx, dy=0.0, dz=dz, max_range_mm=5000.0)
    assert hit is not None
    x, y, z, r, g, b = hit
    expected_t = 100.0 / dx  # solving x0 + t*dx = 100
    expected_z = 800.0 + expected_t * dz
    assert abs(x - 100.0) < 1e-6
    assert abs(z - expected_z) < 1e-6
    assert 0.0 < expected_z < 900.0, "test setup check: should land mid-wall, not at an edge"


def test_wall_wins_over_the_floor_when_both_are_in_range():
    """A ray that would eventually cross the (unbounded) floor plane, but
    hits a nearer wall first, must report the wall — not the floor."""
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))], wall_height_mm=900.0)
    pitch = math.radians(-1.0)  # barely down: floor is very far away along this ray
    dx, dz = math.cos(pitch), math.sin(pitch)
    hit = room._cast_3d(0.0, 0.0, 100.0, dx=dx, dy=0.0, dz=dz, max_range_mm=100000.0)
    assert hit is not None
    x, y, z, r, g, b = hit
    assert abs(x - 100.0) < 1e-6, "must have hit the wall (x=100), not sailed past to the floor"
    assert (r, g, b) == SIMULATED_WALL_COLOR


def test_cast_3d_respects_max_range():
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))], wall_height_mm=900.0)
    hit = room._cast_3d(0.0, 0.0, 0.0, dx=1.0, dy=0.0, dz=0.0, max_range_mm=50.0)
    assert hit is None  # wall is 100mm away, beyond the 50mm cap


# ------------------------------------------------------------- depth_scan


def test_depth_scan_sample_count_matches_h_times_v_when_everything_hits():
    """A small box: every ray, across the whole FOV, hits either a wall or
    the floor — nothing escapes to return None."""
    room = SimulatedRoom.rectangle(4000, 4000, margin_mm=0.0)
    points = room.depth_scan(
        2000.0, 2000.0, 300.0, yaw_deg=0.0, pitch_deg=-20.0,
        h_fov_deg=60.0, v_fov_deg=40.0, h_samples=5, v_samples=4,
        max_range_mm=5000.0,
    )
    assert len(points) == 5 * 4


def test_depth_scan_points_lie_within_the_expected_fov_cone():
    """Every returned point's bearing (from the scan origin) must fall
    within the requested horizontal FOV — a coarse but real geometric
    sanity check that samples aren't leaking outside the cone."""
    room = SimulatedRoom.rectangle(6000, 6000, margin_mm=0.0)
    ox, oy = 3000.0, 3000.0
    points = room.depth_scan(
        ox, oy, 300.0, yaw_deg=0.0, pitch_deg=0.0,
        h_fov_deg=60.0, v_fov_deg=20.0, h_samples=7, v_samples=3,
        max_range_mm=5000.0,
    )
    assert len(points) > 0
    for x, y, z, r, g, b in points:
        bearing = math.degrees(math.atan2(y - oy, x - ox))
        assert -30.5 <= bearing <= 30.5, f"point at bearing {bearing} escaped the +/-30deg FOV"


def test_depth_scan_with_a_single_sample_uses_the_center_angle():
    """h_samples=1/v_samples=1 must not divide by zero (the (n-1) in the
    fraction calc) and must aim exactly at (yaw_deg, pitch_deg)."""
    room = SimulatedRoom(walls=[((100.0, -1000.0), (100.0, 1000.0))], wall_height_mm=900.0)
    points = room.depth_scan(
        0.0, 0.0, 0.0, yaw_deg=0.0, pitch_deg=0.0,
        h_fov_deg=60.0, v_fov_deg=40.0, h_samples=1, v_samples=1,
        max_range_mm=1000.0,
    )
    assert len(points) == 1
    x, y, z, r, g, b = points[0]
    assert abs(x - 100.0) < 1e-6
    assert abs(y - 0.0) < 1e-6
    assert (r, g, b) == SIMULATED_WALL_COLOR  # level ray at z=0 hits the wall, not the floor


def test_depth_scan_colors_wall_and_floor_hits_differently():
    """A wide-enough vertical FOV should pick up both a wall (level rays)
    and the floor (downward-angled rays) in the same scan, each carrying
    its own synthetic colour — see SIMULATED_WALL_COLOR/SIMULATED_FLOOR_COLOR."""
    room = SimulatedRoom(walls=[((300.0, -1000.0), (300.0, 1000.0))], wall_height_mm=900.0)
    points = room.depth_scan(
        0.0, 0.0, 150.0, yaw_deg=0.0, pitch_deg=-20.0,
        h_fov_deg=10.0, v_fov_deg=80.0, h_samples=2, v_samples=9,
        max_range_mm=2000.0,
    )
    colors = {(r, g, b) for _, _, _, r, g, b in points}
    assert SIMULATED_WALL_COLOR in colors
    assert SIMULATED_FLOOR_COLOR in colors


# ------------------------------------------------------------ _ray_box_distance

_TEST_BOX = FurnitureBox.centered(
    x_center=150.0, y_center=0.0, width_x_mm=100.0, depth_y_mm=100.0,
    height_mm=100.0, color=(1, 2, 3),
)  # x:[100,200] y:[-50,50] z:[0,100]


def test_ray_box_distance_hits_the_near_face():
    t = _ray_box_distance(0.0, 0.0, 50.0, dx=1.0, dy=0.0, dz=0.0, box=_TEST_BOX)
    assert t is not None
    assert abs(t - 100.0) < 1e-9


def test_ray_box_distance_misses_a_box_off_to_the_side():
    t = _ray_box_distance(0.0, 200.0, 50.0, dx=1.0, dy=0.0, dz=0.0, box=_TEST_BOX)
    assert t is None


def test_ray_box_distance_box_entirely_behind_the_ray_is_a_miss():
    t = _ray_box_distance(300.0, 0.0, 50.0, dx=1.0, dy=0.0, dz=0.0, box=_TEST_BOX)
    assert t is None


def test_ray_box_distance_ray_origin_inside_the_box_returns_the_exit_point():
    """Standard slab-method behaviour for a ray starting inside a solid:
    t_min comes out negative (the "entry" is behind the origin, since the
    origin is already past it), so the exit point (t_max) is the nearest
    non-negative crossing — the well-known algorithm's own behaviour, not
    an invented special case, but worth a test making it explicit rather
    than surprising."""
    t = _ray_box_distance(150.0, 0.0, 50.0, dx=1.0, dy=0.0, dz=0.0, box=_TEST_BOX)
    assert t is not None
    assert abs(t - 50.0) < 1e-9  # exits at x=200, 50mm from the origin at x=150


# ------------------------------------------------------- furniture in _cast_3d


def test_cast_3d_furniture_wins_over_a_more_distant_wall():
    room = SimulatedRoom(walls=[((1000.0, -1000.0), (1000.0, 1000.0))], wall_height_mm=900.0)
    room.add_furniture(300.0, 0.0, 100.0, 100.0, 200.0, color=(9, 9, 9))
    hit = room._cast_3d(0.0, 0.0, 50.0, dx=1.0, dy=0.0, dz=0.0, max_range_mm=5000.0)
    assert hit is not None
    x, y, z, r, g, b = hit
    assert abs(x - 250.0) < 1e-9  # near face of the furniture box, not the wall at x=1000
    assert (r, g, b) == (9, 9, 9)


def test_cast_3d_wall_wins_over_more_distant_furniture():
    room = SimulatedRoom(walls=[((150.0, -1000.0), (150.0, 1000.0))], wall_height_mm=900.0)
    room.add_furniture(1000.0, 0.0, 100.0, 100.0, 200.0, color=(9, 9, 9))
    hit = room._cast_3d(0.0, 0.0, 50.0, dx=1.0, dy=0.0, dz=0.0, max_range_mm=5000.0)
    assert hit is not None
    x, y, z, r, g, b = hit
    assert abs(x - 150.0) < 1e-9
    assert (r, g, b) == SIMULATED_WALL_COLOR


def test_depth_scan_picks_up_furniture_color():
    room = SimulatedRoom(walls=[((2000.0, -2000.0), (2000.0, 2000.0))])
    room.add_furniture(500.0, 0.0, 200.0, 200.0, 300.0, color=(9, 9, 9))
    points = room.depth_scan(
        0.0, 0.0, 150.0, yaw_deg=0.0, pitch_deg=0.0,
        h_fov_deg=10.0, v_fov_deg=10.0, h_samples=3, v_samples=3,
        max_range_mm=2000.0,
    )
    colors = {(r, g, b) for _, _, _, r, g, b in points}
    assert (9, 9, 9) in colors


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
