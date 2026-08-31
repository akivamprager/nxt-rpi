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
from dataclasses import dataclass, field

Point = tuple[float, float]
Segment = tuple[Point, Point]

#: The simulator has no source camera image to sample real colour from —
#: unlike depth_estimator.make_depth_scanner's real-hardware path, which
#: colorizes each point from the actual frame. These are a synthetic
#: stand-in, not a fabricated *real* colour: matched to scene.html's own
#: wall material (0xe4ddd0) and the average of its procedural wood-floor
#: texture (hsl(30, 32%, ~38%)), so a mesh built from this point cloud
#: looks like the same room the 3D scene view renders, rather than an
#: arbitrary flat grey. Kept as two colours, not one, so wall-hits and
#: floor-hits are visually distinguishable in the exported point
#: cloud/mesh (see _cast_3d, which picks one or the other per hit).
SIMULATED_WALL_COLOR = (228, 221, 208)
SIMULATED_FLOOR_COLOR = (130, 98, 67)


@dataclass(frozen=True)
class FurnitureBox:
    """An axis-aligned box obstacle the depth scanner can hit — visible to
    depth_scan only, deliberately NOT part of clearance()/is_colliding()
    (the robot's actual 2D navigation): furniture here mirrors scene.html's
    own long-standing "dressing the real robot has no awareness of" design
    for the exact same reason scene.html states it — this is a visual/
    mapping-honesty addition (so the point cloud/mesh actually reflects a
    furnished room), not a change to exploration behaviour, which stays
    scoped to what mapping.py/explore.py already handle well."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    color: tuple[int, int, int]

    @classmethod
    def centered(
        cls,
        x_center: float,
        y_center: float,
        width_x_mm: float,
        depth_y_mm: float,
        height_mm: float,
        color: tuple[int, int, int],
        z_min: float = 0.0,
    ) -> "FurnitureBox":
        """The ergonomic constructor: centre position + overall dimensions,
        matching how scene.html's furnish() already positions each piece
        (by centre, not by corner)."""
        return cls(
            x_min=x_center - width_x_mm / 2, x_max=x_center + width_x_mm / 2,
            y_min=y_center - depth_y_mm / 2, y_max=y_center + depth_y_mm / 2,
            z_min=z_min, z_max=z_min + height_mm, color=color,
        )


def _ray_box_distance(
    px: float, py: float, pz: float, dx: float, dy: float, dz: float, box: FurnitureBox
) -> float | None:
    """Standard slab-method ray/AABB intersection: the nearest `t >= 0`
    where the ray enters the box, or None if it misses (or the box is
    entirely behind the ray). Textbook algorithm, not project-specific —
    tested against hand-computed cases in test_sim_world.py the same as
    everything else here."""
    t_min, t_max = -math.inf, math.inf
    for p, d, lo, hi in (
        (px, dx, box.x_min, box.x_max),
        (py, dy, box.y_min, box.y_max),
        (pz, dz, box.z_min, box.z_max),
    ):
        if abs(d) < 1e-9:
            if p < lo or p > hi:
                return None  # parallel to this axis and outside its slab
            continue
        t1, t2 = (lo - p) / d, (hi - p) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    if t_max < 0.0:
        return None  # box is entirely behind the ray's origin
    return t_min if t_min >= 0.0 else t_max  # ray origin starts inside the box


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
    can bump into.

    `wall_height_mm` gives the walls (and the depth-camera model below) real
    3D extent — the 2D ultrasonic ray-casting above doesn't need it at all
    (the turret only yaws, never tilts, so every ultrasonic reading is
    already confined to one horizontal slice regardless of wall height),
    but `depth_scan` does.
    """

    walls: list[Segment]
    wall_height_mm: float = 900.0
    #: See FurnitureBox's own docstring: visible to depth_scan, not to
    #: clearance()/is_colliding() — a mapping/visualization detail, not a
    #: navigation one.
    furniture: list[FurnitureBox] = field(default_factory=list)

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

    def add_furniture(
        self,
        x_center: float,
        y_center: float,
        width_x_mm: float,
        depth_y_mm: float,
        height_mm: float,
        color: tuple[int, int, int],
        z_min: float = 0.0,
    ) -> None:
        """Add a box-shaped furniture obstacle the depth scanner can hit —
        see FurnitureBox's docstring for what this does and doesn't affect."""
        self.furniture.append(
            FurnitureBox.centered(x_center, y_center, width_x_mm, depth_y_mm, height_mm, color, z_min)
        )

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

    def _cast_3d(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        dx: float,
        dy: float,
        dz: float,
        max_range_mm: float,
    ) -> tuple[float, float, float, int, int, int] | None:
        """One 3D ray, tested against wall rectangles and the floor plane
        (z=0); returns the nearest hit point in world mm plus a synthetic
        colour (see SIMULATED_WALL_COLOR/SIMULATED_FLOOR_COLOR), or None.

        Walls are vertical, so a wall's plane is fully determined by its XY
        line — moving in Z never leaves it. That means the *same* parametric
        distance `t` that solves the existing 2D ray-segment intersection
        for `(dx, dy)` (used as-is, NOT renormalized to unit length in 2D —
        see the derivation this depends on: X and Y along the ray only ever
        depend on dx and dy, never dz, so whatever `t` satisfies the 2D
        intersection is already exactly the correct 3D parametric distance)
        also gives the correct height at that point: z + t*dz. Confirmed
        empirically in test_sim_world.py against hand-computed 3D scenarios,
        not just argued here.
        """
        best_t: float | None = None
        best_color: tuple[int, int, int] | None = None

        for wall in self.walls:
            t = _ray_segment_distance(x_mm, y_mm, dx, dy, wall)
            if t is None or t > max_range_mm:
                continue
            hit_z = z_mm + t * dz
            if 0.0 <= hit_z <= self.wall_height_mm:
                if best_t is None or t < best_t:
                    best_t = t
                    best_color = SIMULATED_WALL_COLOR

        # Floor: the z=0 plane. Only relevant for rays actually pointing
        # down from above it — deliberately unbounded in XY (see the
        # module-level note in depth_scan): any ray angled enough to exit
        # through a wall hits that wall at a strictly shorter distance than
        # where the floor plane continues on beyond it, so bounding the
        # floor's extent explicitly would never change the result for rays
        # cast from inside the room, which is the only case this is used for.
        if dz < -1e-9 and z_mm > 0.0:
            t_floor = -z_mm / dz
            if t_floor <= max_range_mm and (best_t is None or t_floor < best_t):
                best_t = t_floor
                best_color = SIMULATED_FLOOR_COLOR

        for box in self.furniture:
            t = _ray_box_distance(x_mm, y_mm, z_mm, dx, dy, dz, box)
            if t is not None and t <= max_range_mm and (best_t is None or t < best_t):
                best_t = t
                best_color = box.color

        if best_t is None:
            return None
        return (
            x_mm + best_t * dx, y_mm + best_t * dy, z_mm + best_t * dz, *best_color
        )

    def depth_scan(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        yaw_deg: float,
        pitch_deg: float,
        h_fov_deg: float = 60.0,
        v_fov_deg: float = 45.0,
        h_samples: int = 12,
        v_samples: int = 8,
        max_range_mm: float = 3000.0,
    ) -> list[tuple[float, float, float, int, int, int]]:
        """A simulated depth-camera scan: casts an `h_samples` x `v_samples`
        grid of rays across the given field of view and returns every
        world-frame (x, y, z, r, g, b) point where one actually hit
        something — see SIMULATED_WALL_COLOR/SIMULATED_FLOOR_COLOR for why
        r/g/b is a synthetic stand-in here, unlike real hardware's colorized
        points sampled from an actual camera frame.

        This is a stand-in for a real depth sensor, not a model of what the
        Pi's actual camera can do — a single Pi Camera Module has no depth
        channel at all. Real geometry from a real camera needs either a
        second (tilting) axis of motion or multi-view structure-from-motion,
        both deferred, separate work (see the plan). What this unlocks now:
        an honestly-earned 3D point cloud in the simulation, built the same
        way the existing 2D occupancy grid is — accumulated from repeated
        scans, not read from the ground-truth room — see mission.py's
        depth_scanner hook and pointcloud.py.

        Yaw 0 faces world +x (matching the rest of this project's bearing
        convention); positive pitch tilts up, matching world +z being up.
        """
        points: list[tuple[float, float, float, int, int, int]] = []
        for j in range(v_samples):
            v_frac = 0.0 if v_samples == 1 else (j / (v_samples - 1)) - 0.5
            pitch = pitch_deg + v_frac * v_fov_deg
            pitch_r = math.radians(pitch)
            cos_pitch, sin_pitch = math.cos(pitch_r), math.sin(pitch_r)

            for i in range(h_samples):
                h_frac = 0.0 if h_samples == 1 else (i / (h_samples - 1)) - 0.5
                yaw_r = math.radians(yaw_deg + h_frac * h_fov_deg)

                dx = cos_pitch * math.cos(yaw_r)
                dy = cos_pitch * math.sin(yaw_r)
                dz = sin_pitch

                hit = self._cast_3d(x_mm, y_mm, z_mm, dx, dy, dz, max_range_mm)
                if hit is not None:
                    points.append(hit)
        return points

    def to_dict(self) -> dict:
        """JSON-serialisable wall list, for the 3D scene viewer.

        This is ground-truth room geometry — useful for a visually complete
        demo scene, but worth being explicit that a real robot has no such
        oracle. It only ever knows what mapping.py's occupancy grid has
        actually observed; see pi/web/scene.html's note on this.
        """
        return {"walls": [[list(a), list(b)] for a, b in self.walls]}
