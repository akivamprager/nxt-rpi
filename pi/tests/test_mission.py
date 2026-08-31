"""Tests for the exploration state machine.

The last test here is the real payoff of this whole session's work: a
simulated robot, in a simulated room, running the actual mission loop that
will eventually drive the real robot — autonomously sweeping, mapping,
planning, and driving until the room is explored, with no hardware involved
at all.
"""

import math
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from scout.mapping import OccupancyGrid  # noqa: E402
from scout.mission import DONE, RECOVERING, SWEEPING, ExplorationMission  # noqa: E402
from scout.robot import Robot  # noqa: E402
from scout.transport import SocketTransport  # noqa: E402
from sim_firmware import make_simulated_pair  # noqa: E402
from sim_world import SimulatedRoom  # noqa: E402


def wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class Harness:
    def __init__(self, room=None, speed_multiplier=1.0):
        self.room = room
        self.speed_multiplier = speed_multiplier

    def __enter__(self):
        self.pi_sock, self.sim = make_simulated_pair(telemetry_period_ms=20, world=self.room)
        if self.speed_multiplier != 1.0:
            self.sim.travel_speed *= self.speed_multiplier
            self.sim.rotate_speed *= self.speed_multiplier
            self.sim.turret_speed *= self.speed_multiplier
        self.robot = Robot(SocketTransport(self.pi_sock), telemetry_period_ms=20)
        self.robot.connect(timeout=3.0)
        self.robot.wait_for_telemetry(timeout=2.0)
        return self

    def __exit__(self, *exc_info):
        self.robot.close()
        self.sim.stop()


def test_sweep_updates_the_grid_from_real_sensor_readings():
    room = SimulatedRoom(walls=[((1000.0, -1000.0), (1000.0, 1000.0))])
    with Harness(room) as h:
        grid = OccupancyGrid(width=40, height=20, cell_size_mm=100.0, origin_y_mm=-1000)
        mission = ExplorationMission(h.robot, grid, sweep_angles=(0.0,))

        mission._do_sweep()

        # Facing +x (heading 0), turret at 0: sensor bearing is 0, wall at 1000mm.
        assert grid.is_free(2, grid.world_to_cell(0, 0)[1])
        echo_cx, echo_cy = grid.world_to_cell(1000, 0)
        assert grid.is_occupied(echo_cx, echo_cy)


def test_recover_backs_away_from_the_obstacle():
    room = SimulatedRoom(walls=[((500.0, -1000.0), (500.0, 1000.0))])
    with Harness(room) as h:
        h.robot.set_safety(False)  # drive all the way to contact, as in the bumper tests
        grid = OccupancyGrid(width=10, height=10, cell_size_mm=100.0)
        mission = ExplorationMission(h.robot, grid)

        h.robot.travel(2000, wait=False)
        assert wait_until(lambda: h.sim.bumpers != 0, timeout=5.0)
        stuck_x = h.sim.x

        mission.state = RECOVERING
        mission._do_recover()

        assert h.sim.x < stuck_x - 50.0, "should have backed away from the wall"
        assert mission.state == SWEEPING


def test_stuck_target_is_avoided_on_the_next_plan():
    """Regression test for a real bug caught via live testing: the turret
    can only sweep +/-120 degrees (TURRET_MAX_ANGLE_DEG), so there is always
    a blind arc directly behind the robot that no amount of sweeping-in-
    place will ever resolve. A frontier cell sitting in that blind arc,
    right next to the robot's own (always-free) cell, is always 'nearest' —
    without the fix in _do_plan, the mission would retry it forever with
    zero progress.
    """
    room = SimulatedRoom(walls=[((2000.0, -2000.0), (2000.0, 2000.0))])  # far away, irrelevant
    with Harness(room) as h:
        grid = OccupancyGrid(width=40, height=40, cell_size_mm=100.0, origin_x_mm=-2000, origin_y_mm=-2000)
        mission = ExplorationMission(h.robot, grid, min_frontier_cluster=1)

        # Robot at (0, 0). Manufacture exactly the stuck scenario: the
        # robot's own cell is free, one cell "behind" it is permanently
        # unknown (as if the turret's blind arc never covered that
        # direction), and — crucially — a second, resolvable frontier
        # exists elsewhere, far enough that it would only be chosen once
        # the stuck one is blacklisted.
        origin_cx, origin_cy = grid.world_to_cell(0, 0)
        grid.log_odds[origin_cy][origin_cx] = -2.0  # free: the robot's own cell
        # its unknown neighbour stays at 0.0 (unknown) by default — the
        # permanently-blind cell "behind" the robot

        far_cx, far_cy = grid.world_to_cell(1000, 0)
        for dx in range(-1, 2):
            grid.log_odds[far_cy][far_cx + dx] = -2.0  # a real, reachable free patch

        # First plan: the stuck (robot's-own-cell) frontier is nearest —
        # distance zero beats anything a thousand mm away.
        mission._do_plan()
        first_target = mission._target
        assert first_target is not None
        # The robot's own 100mm cell center is at most ~71mm (half a cell
        # diagonal) from (0, 0) — comfortably closer than the far cluster
        # at ~1000mm, which is the point of this setup.
        assert math.hypot(first_target[0], first_target[1]) < 150.0, (
            "test setup check: the stuck frontier should be nearest initially"
        )

        # Second plan, robot hasn't moved (as would happen if DRIVING made
        # no real progress toward a target at distance ~0) — this must
        # trigger the blacklist and pick the far, resolvable frontier instead.
        mission._do_plan()
        second_target = mission._target
        assert second_target != first_target, (
            "must not keep re-selecting an unresolvable target with zero movement"
        )
        assert first_target in mission._avoid


def test_stuck_detection_is_not_fooled_by_a_recovery_backup():
    """Regression test for a real bug caught via live testing: the first
    version of the stuck-target fix measured raw robot displacement between
    plans, and RECOVERING's ~150mm backup after every bump always satisfies
    that — so a robot oscillating approach-bump-retreat against the same
    wall forever looked like 'plenty of movement' each cycle and was never
    blacklisted. The user's exact words: 'it just keeps banging into the
    same wall.'

    Simulated here directly via set_pose (not a real bump), because what's
    under test is _do_plan's progress arithmetic, not the drive/recover
    state machine (that part is already covered by test_recover_backs_
    away_from_the_obstacle). The robot moves 150mm between plans — comfortably
    above the OLD 30mm displacement threshold, which would have let this
    slide — but in the wrong direction, so distance to the target does not
    improve. The new check must catch that.
    """
    room = SimulatedRoom(walls=[((2000.0, -2000.0), (2000.0, 2000.0))])
    with Harness(room) as h:
        grid = OccupancyGrid(width=50, height=40, cell_size_mm=100.0, origin_x_mm=-2000, origin_y_mm=-2000)
        mission = ExplorationMission(h.robot, grid, min_frontier_cluster=1)

        # A near "trap" frontier and a far, genuinely reachable one.
        trap_cx, trap_cy = grid.world_to_cell(300, 0)
        for dx in (-1, 0, 1):
            grid.log_odds[trap_cy][trap_cx + dx] = -2.0
        far_cx, far_cy = grid.world_to_cell(1800, 0)
        for dx in (-1, 0, 1):
            grid.log_odds[far_cy][far_cx + dx] = -2.0

        h.robot.set_pose(0, 0, 0)
        assert wait_until(lambda: abs(h.robot.telemetry.x_mm) < 1.0)
        mission._do_plan()
        first_target = mission._target
        assert first_target[0] < 1000, "sanity check: should start by targeting the near trap"

        # "Recovery" moves the robot 150mm — real, substantial displacement,
        # but backward, away from the trap it was just approaching.
        h.robot.set_pose(-150, 0, 0)
        assert wait_until(lambda: abs(h.robot.telemetry.x_mm - (-150)) < 1.0)
        mission._do_plan()

        assert mission._target != first_target, (
            "150mm of real displacement in the wrong direction must still "
            "count as stuck — this is exactly what let the mission bang "
            "into the same wall forever before this fix"
        )
        assert mission._target[0] > 1000, (
            f"should have moved on to the far frontier, got {mission._target}"
        )
        assert first_target in mission._avoid


def test_mission_ends_rather_than_looping_forever_on_a_single_unreachable_frontier():
    """Regression test for a bug in the first version of the stuck-target
    fix itself, caught via live testing: when the ONLY remaining frontier is
    the one just blacklisted, `pick()` used to fall back to considering all
    frontiers again — silently un-blacklisting the very target it had just
    given up on, and immediately re-selecting it. That produced a tight,
    permanent loop with zero progress and zero termination. The fix must
    end the mission instead once nothing un-blacklisted remains.
    """
    room = SimulatedRoom(walls=[((2000.0, -2000.0), (2000.0, 2000.0))])
    with Harness(room) as h:
        grid = OccupancyGrid(width=40, height=40, cell_size_mm=100.0, origin_x_mm=-2000, origin_y_mm=-2000)
        mission = ExplorationMission(h.robot, grid, min_frontier_cluster=1)

        # Exactly one frontier exists: the robot's own cell, per the same
        # setup as the single-target regression test above.
        origin_cx, origin_cy = grid.world_to_cell(0, 0)
        grid.log_odds[origin_cy][origin_cx] = -2.0

        mission._do_plan()  # picks the only frontier there is
        assert mission._target is not None

        mission._do_plan()  # robot hasn't moved: must blacklist it...
        # ...and since nothing else exists, must end rather than loop.
        assert mission.state == DONE


def test_sweep_applies_a_localizer_correction():
    """A stand-in for real vision: the localizer hook is a plain callable,
    so this proves the wiring (sweep -> localizer -> robot.set_pose) works
    without needing opencv or a camera at all. main.py plugs a real
    MarkerLocalizer (vision.py + localize.py) into this exact same hook.
    """
    room = SimulatedRoom(walls=[((2000.0, -2000.0), (2000.0, 2000.0))])
    with Harness(room) as h:
        grid = OccupancyGrid(width=10, height=10, cell_size_mm=100.0)
        calls = []

        def fake_localizer(turret_deg):
            calls.append(turret_deg)
            if turret_deg == 0.0:
                from scout.pose2d import Pose2D

                return Pose2D(500.0, -250.0, 45.0)
            return None

        mission = ExplorationMission(
            h.robot, grid, sweep_angles=(-30.0, 0.0, 30.0), localizer=fake_localizer
        )
        mission._do_sweep()

        assert calls == [-30.0, 0.0, 30.0], "localizer must be called at every sweep stop"
        assert wait_until(
            lambda: h.robot.telemetry is not None
            and abs(h.robot.telemetry.x_mm - 500.0) < 1.0
            and abs(h.robot.telemetry.y_mm - (-250.0)) < 1.0
            and abs(h.robot.telemetry.heading_deg - 45.0) < 0.5
        ), "the localizer's correction must have been applied via robot.set_pose"


def test_sweep_accumulates_depth_scanner_points_into_the_point_cloud():
    """A stand-in for a real depth scan: the depth_scanner hook is a plain
    callable, proving the wiring (sweep -> depth_scanner -> point_cloud)
    without needing sim_world.py's 3D geometry at all. demo_explore.py plugs
    a real SimulatedRoom.depth_scan closure into this exact same hook.
    """
    room = SimulatedRoom(walls=[((2000.0, -2000.0), (2000.0, 2000.0))])
    with Harness(room) as h:
        grid = OccupancyGrid(width=10, height=10, cell_size_mm=100.0)
        calls = []

        def fake_depth_scanner(telemetry):
            calls.append(telemetry.turret_deg)
            return [
                (100.0, 200.0, 300.0, 255, 0, 0),
                (100.0, 200.0, 300.0, 255, 0, 0),
                (400.0, 0.0, 0.0, 0, 255, 0),
            ]

        mission = ExplorationMission(
            h.robot, grid, sweep_angles=(-30.0, 0.0), depth_scanner=fake_depth_scanner
        )
        assert len(mission.point_cloud) == 0

        mission._do_sweep()

        assert calls == [-30.0, 0.0], "depth_scanner must be called at every sweep stop"
        # Each of 2 sweep stops returns 3 points, 2 of which are identical —
        # dedup means 2 distinct points per stop, but both stops return the
        # exact same 3 points here, so the total stays at 2 (not 4).
        assert len(mission.point_cloud) == 2


def test_full_mission_explores_a_small_room_and_terminates():
    # Centered on the origin, NOT anchored at a corner: the robot's odometry
    # always starts at (0, 0), and SimulatedRoom.rectangle(w, h) anchors its
    # box at (0, 0) too — meaning the robot would start touching two walls
    # at once, immediately misreading nearly every direction as occupied.
    room = SimulatedRoom(
        walls=[
            ((-800.0, -800.0), (800.0, -800.0)),
            ((800.0, -800.0), (800.0, 800.0)),
            ((800.0, 800.0), (-800.0, 800.0)),
            ((-800.0, 800.0), (-800.0, -800.0)),
        ]
    )
    # 12x simulated speed: this is a dev-loop test, not a physics benchmark —
    # scaling every rate together changes only wall-clock time, not behavior.
    with Harness(room, speed_multiplier=12.0) as h:
        grid = OccupancyGrid(
            width=20, height=20, cell_size_mm=100.0, origin_x_mm=-1000, origin_y_mm=-1000
        )
        mission = ExplorationMission(h.robot, grid, min_frontier_cluster=2)

        thread = threading.Thread(target=mission.run, daemon=True)
        thread.start()
        thread.join(timeout=90.0)
        mission.stop()
        thread.join(timeout=5.0)

        assert not thread.is_alive(), "mission did not wind down after stop()"
        assert mission.state == DONE or grid.coverage() > 0.3, (
            f"mission ended in state={mission.state} with only "
            f"{grid.coverage():.0%} coverage"
        )
        # The room is fully inside the grid's bounds: a real exploration run
        # should map a meaningful fraction of it, not just the starting cell.
        assert grid.coverage() > 0.15


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
