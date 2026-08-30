"""Tests that a SimulatedRoom, wired into SimulatedNXT, actually behaves like
a robot in a real room end to end: the ultrasonic reads real distances, and
driving into a real wall produces a real bumper stop — without a test ever
setting `sim.range_cm` or `sim.bumpers` by hand.

This is the seam the exploration demo depends on, so it gets its own
end-to-end coverage rather than relying on test_robot.py (which deliberately
tests the protocol/safety layer in isolation from any geometry) or
test_sim_world.py (which tests the geometry in isolation from the simulator).
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from scout import protocol as p  # noqa: E402
from scout.robot import Robot  # noqa: E402
from scout.transport import SocketTransport  # noqa: E402
from sim_firmware import make_simulated_pair  # noqa: E402
from sim_world import SimulatedRoom  # noqa: E402


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class Harness:
    def __init__(self, room):
        self.room = room

    def __enter__(self):
        self.pi_sock, self.sim = make_simulated_pair(telemetry_period_ms=20, world=self.room)
        self.robot = Robot(SocketTransport(self.pi_sock), telemetry_period_ms=20)
        self.robot.connect(timeout=3.0)
        self.robot.wait_for_telemetry(timeout=2.0)
        return self

    def __exit__(self, *exc_info):
        self.robot.close()
        self.sim.stop()


def test_ultrasonic_reads_the_real_distance_to_a_wall():
    room = SimulatedRoom(walls=[((1000.0, -1000.0), (1000.0, 1000.0))])
    with Harness(room) as h:
        assert wait_until(lambda: h.robot.telemetry.has_echo)
        # Robot starts at the origin facing +x (heading 0); wall is 1000mm away.
        assert abs(h.robot.telemetry.range_cm - 100) <= 2


def test_ultrasonic_follows_the_turret_not_just_the_chassis():
    """A wall to the side must show up once the turret points at it, even
    though the chassis itself still faces forward."""
    room = SimulatedRoom(
        walls=[
            ((1000.0, -1000.0), (1000.0, 1000.0)),   # ahead
            ((-1000.0, 490.0), (1000.0, 500.0)),      # to the side (roughly +y)
        ]
    )
    with Harness(room) as h:
        h.robot.turret_to(90.0, timeout=5.0)
        assert wait_until(lambda: abs(h.robot.telemetry.range_cm - 50) <= 2, timeout=3.0)


def test_no_echo_when_facing_open_space():
    # Centered on the origin (not anchored at a corner) so the robot's
    # starting pose of (0, 0) is genuinely in the middle of the room, not
    # sitting against a wall.
    room = SimulatedRoom(
        walls=[
            ((-10000.0, -10000.0), (10000.0, -10000.0)),
            ((10000.0, -10000.0), (10000.0, 10000.0)),
            ((10000.0, 10000.0), (-10000.0, 10000.0)),
            ((-10000.0, 10000.0), (-10000.0, -10000.0)),
        ]
    )
    with Harness(room) as h:
        assert wait_until(lambda: h.robot.telemetry is not None)
        assert not h.robot.telemetry.has_echo


def test_ultrasonic_safety_stops_the_robot_before_it_touches_the_wall():
    """With the turret facing forward (the default) and safety enabled, the
    ultrasonic's ~20cm threshold is what stops the robot — well short of
    actual contact. No test sets range_cm or bumpers by hand here."""
    room = SimulatedRoom(walls=[((500.0, -1000.0), (500.0, 1000.0))])
    with Harness(room) as h:
        events = []
        h.robot.on_event(events.append)

        assert h.robot.travel(2000, wait=False)
        assert wait_until(lambda: p.EV_SAFETY_STOP in events, timeout=5.0)

        stopped_x = h.sim.x
        time.sleep(0.3)
        assert abs(h.sim.x - stopped_x) < 5.0, "must actually stay stopped"
        # Stopped comfortably before contact (wall is at x=500), not grazing it.
        assert 250.0 < stopped_x < 320.0
        assert p.EV_BUMPER not in events, "should never have gotten close enough to touch"


def test_bumper_stops_the_robot_once_ultrasonic_safety_is_disabled():
    """With ultrasonic safety off, nothing stops the robot proactively — it
    must drive until the bumper actually makes contact, and stop there."""
    room = SimulatedRoom(walls=[((500.0, -1000.0), (500.0, 1000.0))])
    with Harness(room) as h:
        h.robot.set_safety(False)
        events = []
        h.robot.on_event(events.append)

        assert h.robot.travel(2000, wait=False)
        assert wait_until(lambda: p.EV_BUMPER in events, timeout=5.0)

        stopped_x = h.sim.x
        time.sleep(0.3)
        assert abs(h.sim.x - stopped_x) < 5.0, "must actually stay stopped at the wall"
        # Bumper trips at robot_radius(90) + bump_contact(60) = 150mm out.
        assert 330.0 < stopped_x < 500.0, "should have gotten much closer than before"


def test_robot_does_not_tunnel_through_a_wall_in_one_step():
    """Regression guard for the _advance() clamp: a single fast integration
    step must not let the chassis cross to the far side of a thin wall.
    Safety disabled so the clamp itself — not the proactive ultrasonic
    stop — is what's actually being exercised."""
    room = SimulatedRoom(walls=[((300.0, -1000.0), (300.0, 1000.0))])
    with Harness(room) as h:
        h.robot.set_safety(False)
        h.robot.travel(5000, wait=False)
        time.sleep(1.0)
        assert h.sim.x < 300.0, "the chassis ended up on the far side of the wall"
        assert h.sim.x > 100.0, "should have stopped near the wall, not far short of it"


def test_reverse_uses_the_wall_behind_it_not_ahead():
    room = SimulatedRoom(walls=[((-200.0, -1000.0), (-200.0, 1000.0))])
    with Harness(room) as h:
        assert h.robot.travel(-2000, timeout=8.0) is False or wait_until(
            lambda: h.sim.x <= -100.0, timeout=3.0
        )
        # It must have stopped near the wall behind it, not driven forever.
        assert h.sim.x > -400.0


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
