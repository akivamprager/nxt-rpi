"""Occupancy grid mapping.

A log-odds occupancy grid built from ultrasonic range readings taken at known
poses. Pure standard library — no numpy, no opencv — so this runs anywhere
Python does, including right now with no camera, no Pi, and no NXT attached.

The one rule that matters for correctness, carried over from the protocol
layer: an ultrasonic reading of "no echo" means *no information*, not *clear*.
Treating it as clear would let the map claim open space where the beam simply
scattered off an angled surface. `update_ray` handles this by carving free
space up to the ultrasonic's known max range and marking nothing at the far
end when there was no echo, rather than marking that endpoint as occupied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Log-odds increments per observation. Symmetric magnitude so a single clean
# reading is already decisive (crosses the threshold below) — our ultrasonic
# model has no simulated noise, so there is nothing to gain from requiring
# several readings before believing the first one.
L_FREE = -0.85
L_OCCUPIED = 0.85

# Log-odds are clamped to these bounds so that a heavily-observed cell isn't
# effectively impossible to revise if the world changes (a chair moves, etc).
L_MIN = -4.0
L_MAX = 4.0

# Thresholds for classifying a cell from its accumulated log-odds.
FREE_THRESHOLD = -0.5
OCCUPIED_THRESHOLD = 0.5

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


@dataclass
class OccupancyGrid:
    """A rectangular occupancy grid anchored at a world-frame origin.

    Cell (0, 0) covers the world-frame square
    [origin_x, origin_x + cell_size_mm) x [origin_y, origin_y + cell_size_mm).
    Coordinates follow the robot's odometry frame: millimetres, with the
    robot starting at (0, 0).
    """

    width: int
    height: int
    cell_size_mm: float
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0

    log_odds: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.log_odds:
            self.log_odds = [[0.0] * self.width for _ in range(self.height)]

    # ------------------------------------------------------------- geometry

    def world_to_cell(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        cx = int(math.floor((x_mm - self.origin_x_mm) / self.cell_size_mm))
        cy = int(math.floor((y_mm - self.origin_y_mm) / self.cell_size_mm))
        return cx, cy

    def cell_center(self, cx: int, cy: int) -> tuple[float, float]:
        x = self.origin_x_mm + (cx + 0.5) * self.cell_size_mm
        y = self.origin_y_mm + (cy + 0.5) * self.cell_size_mm
        return x, y

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    # ---------------------------------------------------------------- state

    def state_at(self, cx: int, cy: int) -> int:
        if not self.in_bounds(cx, cy):
            return UNKNOWN
        value = self.log_odds[cy][cx]
        if value >= OCCUPIED_THRESHOLD:
            return OCCUPIED
        if value <= FREE_THRESHOLD:
            return FREE
        return UNKNOWN

    def is_free(self, cx: int, cy: int) -> bool:
        return self.state_at(cx, cy) == FREE

    def is_occupied(self, cx: int, cy: int) -> bool:
        return self.state_at(cx, cy) == OCCUPIED

    def is_unknown(self, cx: int, cy: int) -> bool:
        return self.state_at(cx, cy) == UNKNOWN

    # -------------------------------------------------------------- updates

    def _apply(self, cx: int, cy: int, delta: float) -> None:
        if not self.in_bounds(cx, cy):
            return
        value = self.log_odds[cy][cx] + delta
        self.log_odds[cy][cx] = max(L_MIN, min(L_MAX, value))

    def update_ray(
        self,
        x_mm: float,
        y_mm: float,
        bearing_deg: float,
        range_cm: int,
        has_echo: bool,
        max_range_cm: float = 255.0,
        cone_half_angle_deg: float = 15.0,
        cone_rays: int = 5,
    ) -> None:
        """Integrate one ultrasonic reading, taken from `(x_mm, y_mm)` at the
        given world-frame bearing.

        If `has_echo` is False, `range_cm` is meaningless (see US_NO_ECHO in
        protocol.py) — carve free space out to `max_range_cm` and mark
        nothing at the end, since scattering off an angled surface is far
        more likely at long range than a genuinely empty room that size.

        A real ultrasonic sensor doesn't return a single infinitely-thin
        ray — it has a real beam width (see the plan's Phase 4 notes), and a
        single ping only tells you "something in this whole cone reflected
        at this range," not which exact angle. Modelling it as one
        mathematical ray leaves angular gaps between sweep readings that
        never get marked free, close to the sensor — which starves
        exploration: a frontier cell can end up sitting in one of those
        gaps immediately next to the robot's own position, forever
        unresolved because no single ray ever crosses it, while always
        being the "nearest" thing left to explore.

        So free space is carved across the whole cone (`cone_rays` sub-rays
        spanning +/- `cone_half_angle_deg`), which is the conservative,
        honest thing to claim — nothing in that whole arc blocked the ping
        before this range. Only the *center* ray's endpoint is marked
        occupied, since the true reflecting surface could be anywhere in the
        cone at that range and claiming the whole arc is occupied would be
        needlessly pessimistic.
        """
        step_mm = self.cell_size_mm / 2.0
        sweep_mm = (range_cm if has_echo else max_range_cm) * 10.0
        steps = max(1, int(sweep_mm / step_mm))

        offsets = (
            [0.0]
            if cone_rays <= 1
            else [
                -cone_half_angle_deg + 2 * cone_half_angle_deg * i / (cone_rays - 1)
                for i in range(cone_rays)
            ]
        )

        visited: set[tuple[int, int]] = set()
        for offset in offsets:
            radians = math.radians(bearing_deg + offset)
            dx, dy = math.cos(radians), math.sin(radians)
            for i in range(steps):
                distance = i * step_mm
                cx, cy = self.world_to_cell(x_mm + dx * distance, y_mm + dy * distance)
                if (cx, cy) not in visited:
                    visited.add((cx, cy))
                    self._apply(cx, cy, L_FREE)

        if has_echo:
            radians = math.radians(bearing_deg)
            end_x = x_mm + math.cos(radians) * sweep_mm
            end_y = y_mm + math.sin(radians) * sweep_mm
            cx, cy = self.world_to_cell(end_x, end_y)
            self._apply(cx, cy, L_OCCUPIED)

    # --------------------------------------------------------------- export

    def snapshot(self) -> list[list[int]]:
        """A UNKNOWN/FREE/OCCUPIED grid, cheap to JSON-encode for a dashboard."""
        return [
            [self.state_at(cx, cy) for cx in range(self.width)]
            for cy in range(self.height)
        ]

    def coverage(self) -> float:
        """Fraction of cells that are no longer unknown."""
        total = self.width * self.height
        if total == 0:
            return 0.0
        known = sum(
            1
            for row in range(self.height)
            for col in range(self.width)
            if self.state_at(col, row) != UNKNOWN
        )
        return known / total
