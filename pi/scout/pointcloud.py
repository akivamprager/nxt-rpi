"""A 3D point cloud accumulated from repeated depth-camera scans.

Mirrors mapping.py's honesty principle for the 2D occupancy grid, one level
up: this is built from repeated scans over time (see mission.py's
`depth_scanner` hook), not read from ground-truth room geometry. In the
simulator, points come from sim_world.SimulatedRoom.depth_scan — a stand-in
for a real depth sensor, since a single Pi Camera Module has no depth
channel of its own. On real hardware, points come from
depth_estimator.make_depth_scanner, colorized by sampling the source camera
frame at each backprojected pixel.

Every point carries a colour: (x, y, z, r, g, b), r/g/b in [0, 255]. The
simulator has no source image to sample, so its points use a fixed neutral
grey (see sim_world.py's SIMULATED_POINT_COLOR) — a placeholder honestly
documented as such, not a fabricated real colour, so the schema stays
uniform end to end (mission.py, /pointcloud.json, scene.html) without a
"colorized or not" branch anywhere downstream.

Pure standard library — testable with synthetic points and no hardware, no
camera, no numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

Point3 = tuple[float, float, float]
Color = tuple[int, int, int]
ColoredPoint = tuple[float, float, float, int, int, int]


@dataclass
class PointCloudMap:
    #: Points are rounded to this resolution (mm) before storing, keyed by
    #: the rounded coordinate — this is what keeps memory and the JSON
    #: payload bounded by the room's surface area at this resolution, not by
    #: how many scans have run. A real depth sensor wouldn't have infinite
    #: precision either, so this isn't purely a memory hack.
    resolution_mm: float = 10.0
    #: A hard backstop against unbounded growth regardless of dedup — cheap
    #: insurance for a public, long-running (SCOUT_LOOP) deployment.
    max_points: int = 200_000

    #: Rounded (x, y, z) -> (r, g, b). A later scan of the same spatial cell
    #: overwrites its colour (last-wins) rather than merging — simplest
    #: correct behaviour, and self-correcting for a transient bad reading
    #: (e.g. lighting flicker) since the next scan just overwrites it again.
    _points: dict[Point3, Color] = field(default_factory=dict, repr=False)

    def add_points(self, points: Iterable[ColoredPoint]) -> None:
        if len(self._points) >= self.max_points:
            return
        r = self.resolution_mm
        for x, y, z, red, green, blue in points:
            if len(self._points) >= self.max_points:
                return
            key = (round(x / r) * r, round(y / r) * r, round(z / r) * r)
            self._points[key] = (red, green, blue)

    def __len__(self) -> int:
        return len(self._points)

    def clear(self) -> None:
        self._points.clear()

    def to_dict(self) -> dict:
        return {
            "points": [
                [x, y, z, r, g, b] for (x, y, z), (r, g, b) in self._points.items()
            ],
            "resolution_mm": self.resolution_mm,
        }
