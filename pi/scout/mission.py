"""The autonomous exploration state machine.

Wires Robot + a turret sweep + OccupancyGrid + explore.py together into the
loop described in the plan: SWEEP -> PLAN -> DRIVE -> (bumper?) RECOVER ->
SWEEP. Runs its own loop, meant to execute on a background thread while a
dashboard polls `snapshot()` for the live state.

One rule that matters for correctness: obstacle events (bumper or ultrasonic
safety stop) do NOT produce an EV_MOVE_DONE — see ScoutServer.java and
sim_firmware.py's `_halt()`. So driving here never waits on Robot.travel()'s
own blocking completion; it starts the move with `wait=False` and polls
telemetry itself, watching for either a normal finish or an obstacle event.
Relying on Robot.travel()'s default wait would block for its full timeout
every time the robot bumps into something.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import explore
from . import protocol as p
from .explore import Frontier
from .mapping import OccupancyGrid
from .pointcloud import PointCloudMap
from .pose2d import Pose2D
from .robot import Robot, RobotError

#: Called with the turret's current bearing after each sweep stop; returns a
#: corrected chassis Pose2D if a known marker was seen from there, or None.
#: Kept as a plain callable (rather than importing vision.py/localize.py
#: directly) so mission.py has no dependency on opencv — a mission with no
#: localizer behaves exactly as before, which is what every existing test
#: exercises. See main.py for how a real MarkerLocalizer plugs in here.
Localizer = Callable[[float], Optional[Pose2D]]

#: Called with the current telemetry after each sweep stop; returns a list
#: of world-frame (x_mm, y_mm, z_mm, r, g, b) coloured points from a
#: depth-camera-like scan taken from there (empty if none). Same pattern as
#: Localizer: mission.py has no dependency on sim_world.py or any particular
#: camera geometry, only on accumulating whatever points a scan produces —
#: see demo_explore.py (simulated) and live_explore.py (real hardware) for
#: how a depth scanner plugs in here.
DepthScanner = Callable[[p.Telemetry], list[tuple[float, float, float, int, int, int]]]

IDLE, SWEEPING, PLANNING, DRIVING, RECOVERING, DONE = (
    "IDLE",
    "SWEEPING",
    "PLANNING",
    "DRIVING",
    "RECOVERING",
    "DONE",
)

#: Turret bearings to sample at each stop. Comfortably inside the firmware's
#: +/-120 degree travel limit (see TURRET_MAX_ANGLE_DEG) with margin to spare.
DEFAULT_SWEEP_ANGLES = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)

#: Cap on how far to drive toward a frontier before stopping to re-sweep.
#: Short hops mean a wrong straight-line choice self-corrects quickly rather
#: than committing to a long drive toward a target that turns out to be
#: behind an obstacle — see explore.nearest_frontier's docstring.
DEFAULT_TRAVEL_STEP_MM = 400.0

#: How far to back away after an obstacle stop, before resuming the mission.
RECOVERY_BACKUP_MM = -150.0


@dataclass
class MissionEvent:
    timestamp: float
    text: str


class ExplorationMission:
    def __init__(
        self,
        robot: Robot,
        grid: OccupancyGrid,
        sweep_angles: tuple[float, ...] = DEFAULT_SWEEP_ANGLES,
        travel_step_mm: float = DEFAULT_TRAVEL_STEP_MM,
        min_frontier_cluster: int = 3,
        localizer: Optional[Localizer] = None,
        depth_scanner: Optional[DepthScanner] = None,
    ) -> None:
        self.robot = robot
        self.grid = grid
        self.sweep_angles = sweep_angles
        self.travel_step_mm = travel_step_mm
        self.min_frontier_cluster = min_frontier_cluster
        self.localizer = localizer
        self.depth_scanner = depth_scanner
        #: Always created (cheap when empty) so callers can read
        #: mission.point_cloud unconditionally, whether or not a
        #: depth_scanner is actually configured.
        self.point_cloud = PointCloudMap()

        self.state = IDLE
        self._stop = threading.Event()
        self._obstacle_hit = threading.Event()
        self._lock = threading.Lock()

        self._target: tuple[float, float] | None = None
        self._frontier_count = 0
        self.log: deque[MissionEvent] = deque(maxlen=100)

        #: Frontiers that repeatedly failed to resolve — see _do_plan's
        #: stuck-target detection. Pruned as the map changes.
        self._avoid: set[tuple[float, float]] = set()
        self._previous_target: tuple[float, float] | None = None
        self._previous_distance_to_target: float | None = None

        robot.on_event(self._on_robot_event)

    # ------------------------------------------------------------- lifecycle

    def stop(self) -> None:
        """Ask the mission loop to end after its current step."""
        self._stop.set()

    def run(self) -> None:
        """The blocking mission loop. Run this on a background thread."""
        self._log("mission started")
        self.state = SWEEPING
        try:
            while not self._stop.is_set():
                if self.state == SWEEPING:
                    self._do_sweep()
                    self.state = PLANNING

                elif self.state == PLANNING:
                    self._do_plan()

                elif self.state == DRIVING:
                    self._do_drive()

                elif self.state == RECOVERING:
                    self._do_recover()

                elif self.state == DONE:
                    self._log("exploration complete: no frontiers left")
                    break
        except RobotError as exc:
            self._log(f"mission aborted: {exc}")
        finally:
            try:
                self.robot.stop()
            except (RobotError, IOError):
                pass
            # Undo the on_event registration from __init__ — robot outlives
            # any one mission (see demo_explore.py's SCOUT_LOOP restart
            # loop), so without this every finished mission stays reachable
            # forever via robot._event_callbacks, leaking its whole point
            # cloud. See Robot.remove_event_callback's docstring.
            self.robot.remove_event_callback(self._on_robot_event)

    # ------------------------------------------------------------- SWEEPING

    def _do_sweep(self) -> None:
        self._log("sweeping")
        for angle in self.sweep_angles:
            if self._stop.is_set():
                return
            self.robot.turret_to(angle, wait=True, timeout=5.0)
            telemetry = self.robot.telemetry
            if telemetry is None:
                continue
            world_bearing = telemetry.heading_deg + telemetry.turret_deg
            self.grid.update_ray(
                x_mm=telemetry.x_mm,
                y_mm=telemetry.y_mm,
                bearing_deg=world_bearing,
                range_cm=telemetry.range_cm,
                has_echo=telemetry.has_echo,
            )

            if self.localizer is not None:
                corrected = self.localizer(telemetry.turret_deg)
                if corrected is not None:
                    self.robot.set_pose(corrected.x_mm, corrected.y_mm, corrected.heading_deg)
                    self._log(
                        f"localized: corrected pose to ({corrected.x_mm:.0f}, "
                        f"{corrected.y_mm:.0f}, {corrected.heading_deg:.1f}deg)"
                    )

            if self.depth_scanner is not None:
                points = self.depth_scanner(telemetry)
                if points:
                    self.point_cloud.add_points(points)
        self.robot.turret_to(0.0, wait=True, timeout=5.0)

    # ------------------------------------------------------------- PLANNING

    #: If the same frontier is chosen again and the robot's distance to it
    #: hasn't shrunk by at least this much since the last time it was
    #: picked, treat it as stuck — see the check below. Must be comfortably
    #: smaller than a normal hop (DEFAULT_TRAVEL_STEP_MM) so real multi-hop
    #: progress is never mistaken for being stuck, and comfortably larger
    #: than RECOVERY_BACKUP_MM so a bump-recover-bump oscillation — which
    #: moves the robot by more than that on every single cycle without ever
    #: actually closing on the target — cannot disguise itself as progress.
    _STUCK_PROGRESS_THRESHOLD_MM = 60.0

    def _do_plan(self) -> None:
        frontiers = explore.find_frontiers(
            self.grid, min_cluster_size=self.min_frontier_cluster
        )
        with self._lock:
            self._frontier_count = len(frontiers)

        telemetry = self.robot.telemetry
        if telemetry is None:
            self.state = SWEEPING
            return
        current = (telemetry.x_mm, telemetry.y_mm)

        def pick(exclude: set[tuple[float, float]]) -> Frontier | None:
            """Nearest frontier not in `exclude`, or None if that empties
            the list. Deliberately does NOT fall back to the unfiltered
            list — a caller asking to exclude the blacklist means exactly
            that, and silently ignoring it is what let a single isolated,
            unreachable frontier defeat the blacklist entirely (caught via
            live testing: the mission spun retrying it forever)."""
            candidates = [
                f for f in frontiers
                if (f.centroid_x_mm, f.centroid_y_mm) not in exclude
            ]
            return explore.nearest_frontier(candidates, *current) if candidates else None

        chosen = pick(self._avoid)
        if chosen is None and self._avoid:
            # Everything left is currently blacklisted. Give them all
            # another chance rather than stopping permanently — the map may
            # have changed enough since they were set aside (a neighbouring
            # cell mapped from a new angle, say) that one is reachable now.
            # If it's genuinely still stuck, the check below will catch it
            # again next cycle.
            self._avoid.clear()
            chosen = pick(self._avoid)

        if chosen is None:
            self.state = DONE
            return
        chosen_point = (chosen.centroid_x_mm, chosen.centroid_y_mm)
        distance_to_target = math.hypot(
            chosen_point[0] - current[0], chosen_point[1] - current[1]
        )

        # A frontier that keeps coming back as "nearest" without the robot
        # actually getting closer to it is not resolvable from here — either
        # it sits in the turret's permanent blind arc directly behind the
        # robot (the firmware caps turret travel at +/-120 degrees, short of
        # the full 360 needed to ever see straight back), or it's on the far
        # side of an obstacle that straight-line nearest-frontier selection
        # has no way to route around (see explore.py's docstring — a full
        # path planner is deliberately out of scope).
        #
        # This checks progress toward the target, not raw robot displacement
        # — the first version of this fix used displacement, and a real bug
        # caught via live testing showed why that's wrong: RECOVERING backs
        # the robot up ~150mm on every bump, which reads as "plenty of
        # movement" even while the robot oscillates approach-bump-retreat
        # against the same wall forever, net distance to the target never
        # actually shrinking. Progress is immune to that: a backup that
        # undoes the approach shows up as near-zero or negative progress,
        # while a real multi-hop approach shows a strong decrease each cycle.
        if self._previous_target == chosen_point and self._previous_distance_to_target is not None:
            progress = self._previous_distance_to_target - distance_to_target
        else:
            progress = float("inf")  # first time targeting this point: no verdict yet

        if self._previous_target == chosen_point and progress < self._STUCK_PROGRESS_THRESHOLD_MM:
            self._avoid.add(chosen_point)
            retry = pick(self._avoid)
            if retry is None:
                # That was the last frontier not already given up on, and
                # it just failed to resolve too. This is as much of the
                # room as this mission can reach without a real path
                # planner — stopping here is honest; looping forever on an
                # unreachable pocket is not.
                self._log("no reachable frontiers remain; ending exploration")
                self.state = DONE
                return
            chosen, chosen_point = retry, (retry.centroid_x_mm, retry.centroid_y_mm)
            distance_to_target = math.hypot(
                chosen_point[0] - current[0], chosen_point[1] - current[1]
            )
            self._log(
                f"target unresolved after {progress:+.0f}mm of progress; trying elsewhere"
            )

        self._previous_target = chosen_point
        self._previous_distance_to_target = distance_to_target
        with self._lock:
            self._target = chosen_point
        self._log(
            f"planning: heading toward ({chosen_point[0]:.0f}, {chosen_point[1]:.0f}), "
            f"{len(frontiers)} frontier(s) known"
        )
        self.state = DRIVING

        # Drop blacklist entries that no longer correspond to a current
        # frontier — the map has changed enough that the concern may not
        # apply if this spot is ever nearest again.
        current_points = {(f.centroid_x_mm, f.centroid_y_mm) for f in frontiers}
        self._avoid &= current_points

    # -------------------------------------------------------------- DRIVING

    def _do_drive(self) -> None:
        telemetry = self.robot.telemetry
        if telemetry is None or self._target is None:
            self.state = SWEEPING
            return

        target_x, target_y = self._target
        dx = target_x - telemetry.x_mm
        dy = target_y - telemetry.y_mm
        distance = math.hypot(dx, dy)
        heading = math.degrees(math.atan2(dy, dx))

        # Turning in place is never inhibited by the safety layer (see
        # ScoutServer.doTurnTo), so this can wait for its full completion.
        self.robot.turn_to(heading, wait=True, timeout=15.0)
        if self._stop.is_set():
            return

        step = min(distance, self.travel_step_mm)
        self._obstacle_hit.clear()

        if not self.robot.travel(step, wait=False):
            # Refused outright — an obstacle is already blocking the way
            # before the robot even started moving.
            self._log("drive refused: obstacle ahead")
            self.state = RECOVERING
            return

        # Poll rather than rely on Robot.travel()'s own wait: a bumper or
        # safety stop halts the firmware's motion but never sends
        # EV_MOVE_DONE, so waiting on that here would block for the full
        # timeout on every single obstacle encountered.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not self._stop.is_set():
            if self._obstacle_hit.is_set():
                self.state = RECOVERING
                return
            current = self.robot.telemetry
            if current is not None and not current.moving:
                break
            time.sleep(0.05)

        self.state = SWEEPING

    # ------------------------------------------------------------ RECOVERING

    def _do_recover(self) -> None:
        self._log("obstacle hit: backing off")
        self._obstacle_hit.clear()
        try:
            self.robot.stop()
            self.robot.travel(RECOVERY_BACKUP_MM, timeout=10.0)
        except RobotError as exc:
            self._log(f"recovery drive failed: {exc}")
        self.state = SWEEPING

    # --------------------------------------------------------------- events

    def _on_robot_event(self, code: int) -> None:
        """Runs on Robot's reader thread — keep this fast, no robot commands."""
        if code in (p.EV_BUMPER, p.EV_SAFETY_STOP):
            self._obstacle_hit.set()

    def _log(self, text: str) -> None:
        with self._lock:
            self.log.append(MissionEvent(time.time(), text))

    # ------------------------------------------------------------- snapshot

    def snapshot(self) -> dict:
        """A JSON-serialisable view of the mission, for a dashboard to poll.

        Deliberately light on locking: this is a visualisation aid, not a
        safety-critical path, and Python's GIL means a reader can only ever
        observe a slightly-stale grid or log, never a torn/corrupt one.
        """
        telemetry = self.robot.telemetry
        with self._lock:
            target = self._target
            frontier_count = self._frontier_count
            log = [{"t": e.timestamp, "text": e.text} for e in list(self.log)[-20:]]

        return {
            "state": self.state,
            "pose": (
                {
                    "x_mm": telemetry.x_mm,
                    "y_mm": telemetry.y_mm,
                    "heading_deg": telemetry.heading_deg,
                    "turret_deg": telemetry.turret_deg,
                }
                if telemetry is not None
                else None
            ),
            "battery_mv": telemetry.battery_mv if telemetry is not None else None,
            "sensors": (
                {
                    "range_cm": telemetry.range_cm,
                    "has_echo": telemetry.has_echo,
                    "color_id": telemetry.color_id,
                    "bumper_pressed": telemetry.bumper_pressed,
                }
                if telemetry is not None
                else None
            ),
            "target": (
                {"x_mm": target[0], "y_mm": target[1]} if target is not None else None
            ),
            "frontier_count": frontier_count,
            "coverage": self.grid.coverage(),
            "grid": {
                "width": self.grid.width,
                "height": self.grid.height,
                "cell_size_mm": self.grid.cell_size_mm,
                "origin_x_mm": self.grid.origin_x_mm,
                "origin_y_mm": self.grid.origin_y_mm,
                "cells": self.grid.snapshot(),
            },
            "log": log,
            "link": self.robot.link_stats(),
        }
