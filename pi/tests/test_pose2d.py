"""Tests for the 2D rigid-pose algebra underneath localize.py.

Pure math, no hardware, no opencv — every case here is checked against a
hand-computed or independently-derivable expected value, not just "does it
run."
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scout.pose2d import IDENTITY, Pose2D, approx_equal, compose, inverse  # noqa: E402


def test_compose_with_identity_is_a_no_op():
    a = Pose2D(123.0, -45.0, 30.0)
    assert approx_equal(compose(IDENTITY, a), a)
    assert approx_equal(compose(a, IDENTITY), a)


def test_compose_pure_translation():
    a = Pose2D(100.0, 0.0, 0.0)
    b = Pose2D(0.0, 50.0, 0.0)
    result = compose(a, b)
    assert approx_equal(result, Pose2D(100.0, 50.0, 0.0))


def test_compose_rotates_the_local_offset_by_the_parent_heading():
    """A local +x offset, composed onto a parent facing +90 degrees (i.e.
    +y), should land along +y — the offset gets rotated with the parent."""
    a = Pose2D(0.0, 0.0, 90.0)
    b = Pose2D(10.0, 0.0, 0.0)
    result = compose(a, b)
    assert approx_equal(result, Pose2D(0.0, 10.0, 90.0))


def test_compose_hand_computed_45_degrees():
    a = Pose2D(0.0, 0.0, 45.0)
    b = Pose2D(10.0, 0.0, 0.0)
    result = compose(a, b)
    expected = 10.0 * math.sqrt(2) / 2
    assert approx_equal(result, Pose2D(expected, expected, 45.0), tol_mm=1e-9)


def test_headings_add_and_wrap():
    a = Pose2D(0.0, 0.0, 170.0)
    b = Pose2D(0.0, 0.0, 30.0)
    result = compose(a, b)
    # 170 + 30 = 200, wrapped to [-180, 180) is -160.
    assert approx_equal(result, Pose2D(0.0, 0.0, -160.0))


def test_inverse_of_identity_is_identity():
    assert approx_equal(inverse(IDENTITY), IDENTITY)


def test_inverse_hand_computed():
    """Worked example from the module docstring's derivation: A = (10, 0,
    90deg) must invert to (0, 10, -90deg)."""
    a = Pose2D(10.0, 0.0, 90.0)
    result = inverse(a)
    assert approx_equal(result, Pose2D(0.0, 10.0, -90.0))


def test_compose_with_inverse_cancels_both_directions():
    """This is the property the whole localization chain in localize.py
    depends on: inverse() must genuinely undo compose() from either side."""
    cases = [
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(100.0, -200.0, 45.0),
        Pose2D(-500.0, 300.0, 179.0),
        Pose2D(1.0, 1.0, -179.5),
        Pose2D(0.0, 0.0, 90.0),
    ]
    for a in cases:
        assert approx_equal(compose(a, inverse(a)), IDENTITY, tol_mm=1e-6), a
        assert approx_equal(compose(inverse(a), a), IDENTITY, tol_mm=1e-6), a


def test_inverse_is_its_own_inverse():
    a = Pose2D(37.0, -84.0, 62.0)
    assert approx_equal(inverse(inverse(a)), a)


def test_compose_is_associative_for_chained_transforms():
    """Required for the multi-hop localization chain (marker -> camera ->
    turret -> chassis) to give the same answer regardless of grouping."""
    a = Pose2D(50.0, 10.0, 20.0)
    b = Pose2D(-30.0, 5.0, 90.0)
    c = Pose2D(15.0, -15.0, -45.0)
    left = compose(compose(a, b), c)
    right = compose(a, compose(b, c))
    assert approx_equal(left, right, tol_mm=1e-9)


def test_approx_equal_treats_180_and_negative_180_as_the_same_heading():
    assert approx_equal(Pose2D(0, 0, 180.0), Pose2D(0, 0, -180.0))


def test_normalized_wraps_heading_into_range():
    assert Pose2D(0, 0, 200.0).normalized().heading_deg == -160.0
    assert Pose2D(0, 0, -200.0).normalized().heading_deg == 160.0


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
