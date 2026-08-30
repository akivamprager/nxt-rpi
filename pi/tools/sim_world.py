"""A simple 2D room for the firmware simulator to sense against.

sim_firmware.py's SimulatedNXT models motion but had no geometry to bump into
or scan — range_cm was whatever a test set it to. This module gives it a
real room: wall segments the ultrasonic ray-casts against and can physically
collide with, so mapping.py and explore.py have something genuine to chew on
without any hardware at all.

Pure standard library. Units are millimetres, matching the rest of the
odometry frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _ray_segment_distance(
    px: float, py: float, dx: float, dy: float, segment: Segment
) -> float | None:
    """Distance from `(px, py)` to `segment` along the unit direction `(dx, dy)`.

    None if the ray (t >= 0) does not hit the segment (0 <= u <= 1).
    Standard parametric line intersection via cross products:
      P + t*D = A + u*(B - A)
    """
    (ax, ay), (bx, by) = segment
    sx, sy = bx - ax, by - ay

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None  # parallel

    t = ((ax - px) * sy - (ay - py) * sx) / denom
    u = ((ax - px) * dy - (ay - py) * dx) / denom

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def _point_segment_distance(px: float, py: float, segment: Segment) -> float:
    (ax, ay), (bx, by) = segment
    sx, sy = bx - ax, by - ay
    length_sq = sx * sx + sy * sy
    if length_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * sx + (py - ay) * sy) / length_sq))
    proj_x, proj_y = ax + t * sx, ay + t * sy
    return math.hypot(px - proj_x, py - proj_y)


@dataclass
class SimulatedRoom:
    """A set of wall segments (mm) that the ultrasonic can see and the robot
    can bump into."""

    walls: list[Segment]

    @classmethod
    def rectangle(cls, width_mm: float, height_mm: float, margin_mm: float = 0.0) -> "SimulatedRoom":
        """A rectangular room from (margin, margin) to (width-margin, height-margin)."""
        x0, y0 = margin_mm, margin_mm
        x1, y1 = width_mm - margin_mm, height_mm - margin_mm
        return cls(
            walls=[
                ((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0)),
            ]
        )

    def add_wall(self, a: Point, b: Point) -> None:
        """Add an interior wall — e.g. a divider, so exploration has to go
        around something rather than just mapping an empty box."""
        self.walls.append((a, b))

    def cast_mm(
        self, x_mm: float, y_mm: float, bearing_deg: float, max_range_mm: float = 1.0e6
    ) -> float | None:
        """Raw ray-cast distance in mm, or None if nothing is within range.

        This is the primitive both `sense()` (cm, rounded, for the ultrasonic)
        and the simulator's motion clamping (mm, exact, for not letting a
        single integration step tunnel through a wall) are built on.
        """
        radians = math.radians(bearing_deg)
        dx, dy = math.cos(radians), math.sin(radians)

        best_mm: float | None = None
        for wall in self.walls:
            distance = _ray_segment_distance(x_mm, y_mm, dx, dy, wall)
            if distance is not None and (best_mm is None or distance < best_mm):
                best_mm = distance

        if best_mm is None or best_mm > max_range_mm:
            return None
        return best_mm

    def sense(
        self, x_mm: float, y_mm: float, bearing_deg: float, max_range_cm: float = 255.0
    ) -> tuple[int, bool]:
        """Ray-cast from `(x_mm, y_mm)` toward `bearing_deg` (world frame degrees).

        Returns (range_cm, has_echo), mirroring the real ultrasonic sensor's
        semantics: has_echo is False, and range_cm meaningless, when nothing
        is in range (matches protocol.US_NO_ECHO upstream).
        """
        best_mm = self.cast_mm(x_mm, y_mm, bearing_deg, max_range_mm=max_range_cm * 10.0)
        if best_mm is None:
            return 255, False
        return int(round(best_mm / 10.0)), True

    def clearance(self, x_mm: float, y_mm: float) -> float:
        """Distance in mm from `(x_mm, y_mm)` to the nearest wall — used for
        collision/bumper checks, not the ultrasonic (which is directional)."""
        return min(_point_segment_distance(x_mm, y_mm, wall) for wall in self.walls)

    def is_colliding(self, x_mm: float, y_mm: float, robot_radius_mm: float = 90.0) -> bool:
        return self.clearance(x_mm, y_mm) < robot_radius_mm

    def to_dict(self) -> dict:
        """JSON-serialisable wall list, for the 3D scene viewer.

        This is ground-truth room geometry — useful for a visually complete
        demo scene, but worth being explicit that a real robot has no such
        oracle. It only ever knows what mapping.py's occupancy grid has
        actually observed; see pi/web/scene.html's note on this.
        """
        return {"walls": [[list(a), list(b)] for a, b in self.walls]}
