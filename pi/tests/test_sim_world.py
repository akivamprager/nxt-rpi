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

from sim_world import SimulatedRoom  # noqa: E402


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
