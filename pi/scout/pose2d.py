"""2D rigid-pose algebra: compose and invert (x, y, heading) transforms.

This is the reusable primitive underneath localize.py's marker-based
localization chain (camera -> turret -> chassis, marker -> camera). It has
no dependency on opencv, cameras, or anything hardware-related — it is pure
geometry, so unlike the vision code that will eventually feed it real data,
it can be fully built and tested before any hardware exists.

Convention
----------
A Pose2D(x, y, heading_deg) represents a rigid transform from a "local"
frame into a "parent" frame: a point (px, py) expressed in the local frame
maps into the parent frame as::

    parent_x = x + px * cos(h) - py * sin(h)
    parent_y = y + px * sin(h) + py * cos(h)

`compose(a, b)` reads as "b's pose, expressed in a's parent frame" — i.e.
b is a transform local to a, and the result is the combined transform from
b's frame all the way out to a's parent frame. `inverse(a)` is the transform
that undoes a: `compose(a, inverse(a))` and `compose(inverse(a), a)` are
both the identity pose, which is exactly what the tests check.

Headings are degrees throughout, matching protocol.py and mapping.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x_mm: float
    y_mm: float
    heading_deg: float

    def normalized(self) -> "Pose2D":
        return Pose2D(self.x_mm, self.y_mm, _normalize_deg(self.heading_deg))


IDENTITY = Pose2D(0.0, 0.0, 0.0)


def _normalize_deg(degrees: float) -> float:
    """Wrap into [-180, 180), matching Protocol.normaliseDegrees in Java."""
    while degrees >= 180.0:
        degrees -= 360.0
    while degrees < -180.0:
        degrees += 360.0
    return degrees


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """b's pose, treated as local to a, expressed in a's parent frame."""
    radians = math.radians(a.heading_deg)
    cos_h, sin_h = math.cos(radians), math.sin(radians)
    return Pose2D(
        x_mm=a.x_mm + b.x_mm * cos_h - b.y_mm * sin_h,
        y_mm=a.y_mm + b.x_mm * sin_h + b.y_mm * cos_h,
        heading_deg=_normalize_deg(a.heading_deg + b.heading_deg),
    )


def inverse(a: Pose2D) -> Pose2D:
    """The transform that undoes `a`."""
    radians = math.radians(a.heading_deg)
    cos_h, sin_h = math.cos(radians), math.sin(radians)
    return Pose2D(
        x_mm=-a.x_mm * cos_h - a.y_mm * sin_h,
        y_mm=a.x_mm * sin_h - a.y_mm * cos_h,
        heading_deg=_normalize_deg(-a.heading_deg),
    )


def approx_equal(a: Pose2D, b: Pose2D, tol_mm: float = 1e-6, tol_deg: float = 1e-6) -> bool:
    """Compares headings modulo 360 — -180 and 180 are the same heading."""
    return (
        abs(a.x_mm - b.x_mm) < tol_mm
        and abs(a.y_mm - b.y_mm) < tol_mm
        and abs(_normalize_deg(a.heading_deg - b.heading_deg)) < tol_deg
    )
