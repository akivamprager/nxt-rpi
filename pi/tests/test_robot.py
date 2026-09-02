"""End-to-end tests of the Pi stack against the firmware simulator.

Exercises transport -> protocol -> Robot with no hardware attached. The safety
tests matter most: they assert the behaviour that keeps the real robot from
driving into walls, and they are the hardest to test with a brick in hand.

Run with:  python3 pi/tests/test_robot.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from scout import protocol as p  # noqa: E402
from scout.robot import Robot, RobotError  # noqa: E402
from scout.transport import SocketTransport  # noqa: E402
from sim_firmware import make_simulated_pair  # noqa: E402


class Harness:
    """A connected Robot plus the simulator it is talking to."""

    def __enter__(self):
        self.pi_sock, self.sim = make_simulated_pair(telemetry_period_ms=20)
        self.robot = Robot(SocketTransport(self.pi_sock), telemetry_period_ms=20)
        self.robot.connect(timeout=3.0)
        self.robot.wait_for_telemetry(timeout=2.0)
        return self

    def __exit__(self, *exc_info):
        self.robot.close()
        self.sim.stop()


def wait_until(predicate, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_connect_and_ping():
    with Harness() as h:
        assert h.robot.ping(timeout=2.0)
        assert h.robot.connected


def test_telemetry_streams():
    with Harness() as h:
        first = h.robot.telemetry
        assert first is not None
        assert wait_until(lambda: h.robot.telemetry.seq != first.seq)
        assert h.robot.telemetry_age < 1.0


def test_travel_moves_and_reports_completion():
    with Harness() as h:
        assert h.robot.travel(300, timeout=10.0)
        assert wait_until(lambda: abs(h.sim.x - 300) < 1.0)
        assert abs(h.robot.telemetry.x_mm - 300) < 5.0


def test_turn_to_absolute_heading():
    with Harness() as h:
        assert h.robot.turn_to(90.0, timeout=10.0)
        assert abs(_normalise(h.sim.heading - 90.0)) < 1.0


def test_turn_takes_the_short_way_around():
    """Turning from +170 to -170 is a 20 degree move, not 340."""
    with Harness() as h:
        h.robot.set_pose(0, 0, 170.0)
        assert wait_until(lambda: abs(h.sim.heading - 170.0) < 0.1)
        start = time.monotonic()
        assert h.robot.turn_to(-170.0, timeout=10.0)
        elapsed = time.monotonic() - start
        # At 60 deg/s, 20 degrees takes ~0.33s and 340 degrees ~5.7s.
        assert elapsed < 2.0, f"took {elapsed:.2f}s; it went the long way round"
        assert abs(_normalise(h.sim.heading - (-170.0))) < 1.0


def test_set_pose_corrects_odometry():
    """This is the mechanism that applies an ArUco fix to the firmware."""
    with Harness() as h:
        h.robot.travel(200, timeout=10.0)
        h.robot.set_pose(1000, -500, 45.0)
        assert wait_until(
            lambda: h.robot.telemetry is not None
            and abs(h.robot.telemetry.x_mm - 1000) < 1.0
            and abs(h.robot.telemetry.y_mm - (-500)) < 1.0
            and abs(h.robot.telemetry.heading_deg - 45.0) < 0.5
        )


def test_forward_is_refused_when_an_obstacle_is_close():
    with Harness() as h:
        h.sim.range_cm = 10  # inside the 20 cm safety threshold
        assert h.robot.travel(500, wait=False) is False, "should be refused"
        time.sleep(0.2)
        assert abs(h.sim.x) < 1.0, "the robot must not have moved"


def test_reverse_is_allowed_while_blocked():
    """Blocking reverse would strand the robot against whatever it just hit."""
    with Harness() as h:
        h.sim.range_cm = 5
        assert h.robot.travel(-200, timeout=10.0)
        assert h.sim.x < -190


def test_safety_stops_a_move_already_in_progress():
    with Harness() as h:
        events = []
        h.robot.on_event(events.append)

        assert h.robot.travel(2000, wait=False)
        assert wait_until(lambda: h.sim.x > 20, timeout=3.0), "should get moving"

        h.sim.range_cm = 5  # an obstacle appears mid-drive
        assert wait_until(lambda: p.EV_SAFETY_STOP in events, timeout=3.0)

        stopped_at = h.sim.x
        time.sleep(0.3)
        assert abs(h.sim.x - stopped_at) < 1.0, "must stay stopped"


def test_bumper_stops_the_robot():
    with Harness() as h:
        events = []
        h.robot.on_event(events.append)

        assert h.robot.travel(2000, wait=False)
        assert wait_until(lambda: h.sim.x > 20, timeout=3.0)

        h.sim.bumpers = 0b01
        assert wait_until(lambda: p.EV_BUMPER in events, timeout=3.0)
        assert h.robot.telemetry.bumper_pressed


def test_bumper_stops_even_with_safety_disabled():
    """Disabling the ultrasonic safety must not disable the bumpers."""
    with Harness() as h:
        h.robot.set_safety(False)
        events = []
        h.robot.on_event(events.append)

        assert h.robot.travel(2000, wait=False)
        assert wait_until(lambda: h.sim.x > 20, timeout=3.0)

        h.sim.bumpers = 0b10
        assert wait_until(lambda: p.EV_BUMPER in events, timeout=3.0)


def test_remove_event_callback_stops_future_delivery():
    """A caller whose lifetime is shorter than the Robot's own (e.g. a
    mission recreated on every SCOUT_LOOP restart) must be able to fully
    unregister — otherwise it stays reachable forever via
    Robot._event_callbacks, which is exactly what leaked memory on the
    public demo until Robot.remove_event_callback existed."""
    with Harness() as h:
        events = []
        h.robot.on_event(events.append)
        h.robot.remove_event_callback(events.append)

        assert h.robot.travel(2000, wait=False)
        assert wait_until(lambda: h.sim.x > 20, timeout=3.0)
        h.sim.bumpers = 0b01
        time.sleep(0.3)
        assert events == []


def test_remove_event_callback_is_a_noop_if_not_registered():
    with Harness() as h:
        h.robot.remove_event_callback(lambda code: None)  # must not raise


def test_no_echo_does_not_trip_safety():
    """255 means 'no information'. Treating it as an obstacle would freeze the
    robot every time the beam scattered off an angled wall."""
    with Harness() as h:
        h.sim.range_cm = p.US_NO_ECHO
        assert h.robot.travel(300, timeout=10.0), "no-echo must not block motion"
        assert h.robot.telemetry.has_echo is False


def test_disabling_safety_permits_close_approach():
    with Harness() as h:
        h.sim.range_cm = 10
        assert h.robot.travel(200, wait=False) is False
        h.robot.set_safety(False)
        assert h.robot.travel(200, timeout=10.0), "should be allowed once disabled"


def test_turret_clamps_to_protect_the_camera_ribbon():
    with Harness() as h:
        h.robot.turret_to(400.0, wait=False)  # far beyond the +/-120 limit
        assert wait_until(lambda: abs(h.sim.turret - 120.0) < 0.5, timeout=6.0)

        h.robot.turret_to(-400.0, wait=False)
        assert wait_until(lambda: abs(h.sim.turret - (-120.0)) < 0.5, timeout=8.0)


def test_turret_reports_actual_bearing_not_requested():
    with Harness() as h:
        h.robot.turret_to(90.0, timeout=6.0)
        assert wait_until(lambda: abs(h.robot.telemetry.turret_deg - 90.0) < 1.0)


def test_turret_to_does_not_return_before_the_turret_actually_moves():
    """Regression test for a real race caught via live testing (in
    mission.py's sweep, which calls turret_to(wait=True) repeatedly): if the
    turret was already stationary, the first telemetry frame polled by
    _wait_until could be a STALE one from before the command was even sent,
    with FLAG_TURRET_MOVING clear for the OLD reason (hadn't started) rather
    than the new one (already arrived) — turret_to(wait=True) would then
    return success immediately while the simulated turret was still at its
    old angle. Several consecutive commands make the race deterministic to
    reproduce, since it depends on winning a narrow timing window.
    """
    with Harness() as h:
        for target in (30.0, -30.0, 60.0, -60.0, 0.0):
            assert h.robot.turret_to(target, timeout=6.0)
            # The instant turret_to() returns, the simulated ground truth
            # must already reflect it — not "will get there eventually."
            assert abs(h.sim.turret - target) < 1.0, (
                f"turret_to({target}) returned before the turret arrived: "
                f"sim.turret={h.sim.turret}"
            )


def test_stop_halts_continuous_drive():
    with Harness() as h:
        assert h.robot.drive(200, 0)
        assert wait_until(lambda: h.sim.x > 20, timeout=3.0)
        h.robot.stop()
        time.sleep(0.1)
        stopped_at = h.sim.x
        time.sleep(0.3)
        assert abs(h.sim.x - stopped_at) < 1.0


def test_link_stats_are_clean_on_a_healthy_link():
    with Harness() as h:
        wait_until(lambda: h.robot.frames_received > 10, timeout=3.0)
        stats = h.robot.link_stats()
        assert stats["checksum_errors"] == 0
        assert stats["frames_received"] > 10
        assert stats["connected"] is True


def test_commands_fail_cleanly_after_the_link_drops():
    h = Harness().__enter__()
    h.sim.stop()
    h.pi_sock.close()
    try:
        for _ in range(20):
            try:
                h.robot.stop()
            except RobotError:
                return  # the expected outcome
            time.sleep(0.05)
        raise AssertionError("expected RobotError once the link was gone")
    finally:
        h.robot.close()


def test_context_manager_stops_the_robot_on_exit():
    pi_sock, sim = make_simulated_pair(telemetry_period_ms=20)
    with Robot(SocketTransport(pi_sock), telemetry_period_ms=20) as robot:
        robot.wait_for_telemetry(timeout=2.0)
        robot.drive(200, 0)
        assert wait_until(lambda: sim.x > 10, timeout=3.0)
    time.sleep(0.2)
    stopped_at = sim.x
    time.sleep(0.3)
    assert abs(sim.x - stopped_at) < 1.0, "exiting the block must stop the motors"
    sim.stop()


def _normalise(degrees):
    while degrees >= 180.0:
        degrees -= 360.0
    while degrees < -180.0:
        degrees += 360.0
    return degrees


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
