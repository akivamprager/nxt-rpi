"""A 3D point cloud accumulated from repeated depth-camera scans.

Mirrors mapping.py's honesty principle for the 2D occupancy grid, one level
up: this is built from repeated scans over time (see mission.py's
`depth_scanner` hook), not read from ground-truth room geometry. In the
simulator, points come from sim_world.SimulatedRoom.depth_scan — a stand-in
for a real depth sensor, since a single Pi Camera Module has no depth
channel of its own (see that method's docstring for what bridging this to
real hardware actually requires).

Pure standard library, same as mapping.py — testable with synthetic points
and no hardware, no camera, no numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

Point3 = tuple[float, float, float]


@dataclass
class PointCloudMap:
    #: Points are rounded to this resolution (mm) before storing, in a set
    #: keyed by the rounded coordinate — this is what keeps memory and the
    #: JSON payload bounded by the room's surface area at this resolution,
    #: not by how many scans have run. A real depth sensor wouldn't have
    #: infinite precision either, so this isn't purely a memory hack.
    resolution_mm: float = 20.0
    #: A hard backstop against unbounded growth regardless of dedup — cheap
    #: insurance for a public, long-running (SCOUT_LOOP) deployment.
    max_points: int = 200_000

    _points: set[Point3] = field(default_factory=set, repr=False)

    def add_points(self, points: Iterable[tuple[float, float, float]]) -> None:
        if len(self._points) >= self.max_points:
            return
        r = self.resolution_mm
        for x, y, z in points:
            if len(self._points) >= self.max_points:
                return
            self._points.add((round(x / r) * r, round(y / r) * r, round(z / r) * r))

    def __len__(self) -> int:
        return len(self._points)

    def clear(self) -> None:
        self._points.clear()

    def to_dict(self) -> dict:
        return {"points": [list(p) for p in self._points], "resolution_mm": self.resolution_mm}
