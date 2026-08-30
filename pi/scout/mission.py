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

from . import explore
from . import protocol as p
from .explore import Frontier
from .mapping import OccupancyGrid
from .robot import Robot, RobotError

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
    ) -> None:
        self.robot = robot
        self.grid = grid
        self.sweep_angles = sweep_angles
        self.travel_step_mm = travel_step_mm
        self.min_frontier_cluster = min_frontier_cluster

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
        self._previous_position: tuple[float, float] | None = None

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
        self.robot.turret_to(0.0, wait=True, timeout=5.0)

    # ------------------------------------------------------------- PLANNING

    #: If the same frontier is chosen twice in a row with the robot having
    #: moved less than this since the last plan, sweeping-in-place isn't
    #: resolving it — see the stuck-target check below.
    _STUCK_MOVEMENT_THRESHOLD_MM = 30.0

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

        # A frontier that keeps coming back as "nearest" while the robot
        # hasn't actually moved is not resolvable by sweeping from here —
        # most likely it sits in the turret's permanent blind arc directly
        # behind the robot (the firmware caps turret travel at +/-120
        # degrees, short of the full 360 needed to ever see straight back),
        # or it's on the far side of an obstacle that straight-line nearest-
        # frontier selection has no way to route around (see explore.py's
        # docstring — a full path planner is deliberately out of scope).
        # Retrying it forever would stall the mission, so set it aside.
        if self._previous_position is None:
            distance_since_last_plan = float("inf")
        else:
            distance_since_last_plan = math.hypot(
                current[0] - self._previous_position[0],
                current[1] - self._previous_position[1],
            )

        if (
            self._previous_target == chosen_point
            and distance_since_last_plan < self._STUCK_MOVEMENT_THRESHOLD_MM
        ):
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
            self._log(
                f"target ({chosen_point[0]:.0f}, {chosen_point[1]:.0f}) unresolved "
                f"after sweeping in place; trying elsewhere"
            )

        self._previous_target = chosen_point
        self._previous_position = current
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
