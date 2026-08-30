"""Frontier-based exploration over an OccupancyGrid.

A frontier cell is a known-free cell touching at least one unknown cell —
the boundary between "we've looked here" and "we haven't yet." Exploring
greedily toward the nearest frontier cluster is a small, well-understood
algorithm that reliably covers a room without needing a full path planner,
which is why it's the one this project uses (see the plan's Phase 4).

Pure standard library, same as mapping.py — testable with a synthetic grid
and no hardware at all.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .mapping import OccupancyGrid

#: 4-connected neighbours are used throughout. 8-connected would treat
#: diagonally-touching cells as adjacent, which tends to merge frontiers that
#: are really on opposite sides of a thin wall.
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Frontier:
    """A cluster of adjacent frontier cells, collapsed to a driving target."""

    cells: tuple[tuple[int, int], ...]
    centroid_x_mm: float
    centroid_y_mm: float
    size: int


def find_frontier_cells(grid: OccupancyGrid) -> set[tuple[int, int]]:
    """Every free cell adjacent to at least one unknown cell.

    Only neighbours *within the grid* count. The grid represents a bounded
    area sized to cover the room with margin, so the edge of the array is a
    modelling boundary, not an unmapped frontier — treating it as one would
    have the robot forever trying to explore past the edge of its own data
    structure.
    """
    frontier: set[tuple[int, int]] = set()
    for cy in range(grid.height):
        for cx in range(grid.width):
            if not grid.is_free(cx, cy):
                continue
            for dx, dy in _NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if grid.in_bounds(nx, ny) and grid.is_unknown(nx, ny):
                    frontier.add((cx, cy))
                    break
    return frontier


def cluster_frontiers(grid: OccupancyGrid, cells: set[tuple[int, int]]) -> list[Frontier]:
    """Group frontier cells into connected clusters via flood fill.

    Clustering matters because a long wall bordering unexplored space
    produces dozens of individual frontier cells; treating each as a separate
    goal would make the robot twitch between neighbours instead of driving
    toward the frontier as a whole.
    """
    remaining = set(cells)
    clusters: list[Frontier] = []

    while remaining:
        start = next(iter(remaining))
        remaining.discard(start)
        component = [start]
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in _NEIGHBORS:
                neighbor = (cx + dx, cy + dy)
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)

        sum_x = sum_y = 0.0
        for cx, cy in component:
            wx, wy = grid.cell_center(cx, cy)
            sum_x += wx
            sum_y += wy
        n = len(component)
        clusters.append(
            Frontier(
                cells=tuple(component),
                centroid_x_mm=sum_x / n,
                centroid_y_mm=sum_y / n,
                size=n,
            )
        )

    return clusters


def find_frontiers(
    grid: OccupancyGrid, min_cluster_size: int = 2
) -> list[Frontier]:
    """Frontier clusters, largest first, with tiny single-cell noise dropped.

    A one-cell frontier is often a stray reading rather than a real unexplored
    opening; requiring at least `min_cluster_size` cells filters most of that
    out without needing anything fancier.
    """
    cells = find_frontier_cells(grid)
    clusters = cluster_frontiers(grid, cells)
    clusters = [c for c in clusters if c.size >= min_cluster_size]
    return sorted(clusters, key=lambda c: c.size, reverse=True)


def nearest_frontier(
    frontiers: list[Frontier], x_mm: float, y_mm: float
) -> Frontier | None:
    """The frontier whose centroid is closest to `(x_mm, y_mm)`.

    Straight-line distance, not a path length around obstacles — this project
    deliberately skips a full path planner (see Phase 4 in the plan) and
    instead re-sweeps and re-plans after every short drive, so a wrong
    straight-line choice self-corrects on the next cycle rather than needing
    to be right up front.
    """
    if not frontiers:
        return None
    return min(
        frontiers,
        key=lambda f: math.hypot(f.centroid_x_mm - x_mm, f.centroid_y_mm - y_mm),
    )
